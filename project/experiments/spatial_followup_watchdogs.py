"""Ordered independent watchdogs: WM update + GRPO, or one baseline GRPO fallback."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from queue_spatial_wm_update import (
    arm as arm_wm_update,
)
from queue_spatial_wm_update import (
    completed_step,
    process_alive,
    release_predecessor,
    sha256,
    write_json,
)
from spatial_wovr_memory_guard import checkpoint_complete
from stop_spatial_grpo import read_process, signal_process, snapshot

PROJECT = Path(__file__).resolve().parents[1]
CONTROL_FILES = [
    "spatial_followup_watchdogs.py",
    "spatial_wovr_pair_launch.py",
    "run_spatial_wovr_8gpu.sh",
    "run_spatial_wovr_20step.sh",
    "run_spatial_wovr_shared_pool_8gpu.sh",
]
ROLES = ("primary", "fallback")


def utc() -> str:
    return datetime.now(UTC).isoformat()


def state(plan: dict, role: str, status: str, **details) -> None:
    value = {
        "role": role,
        "status": status,
        "pid": os.getpid(),
        "updated_utc": utc(),
        **details,
    }
    output = Path(plan["output"])
    write_json(output / f"{role}_state.json", value)
    with (output / f"{role}_events.jsonl").open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
    print(json.dumps(value, sort_keys=True), flush=True)


def verify(plan: dict) -> None:
    if Path("/proc/sys/kernel/random/boot_id").read_text().strip() != plan["boot_id"]:
        raise RuntimeError("This chain belongs to a different execution node/boot")
    if plan["control_code"] != {
        name: sha256(PROJECT / "experiments" / name) for name in CONTROL_FILES
    }:
        raise RuntimeError("Follow-up control code changed after arming")
    if sha256(Path(plan["source_config"])) != plan["source_config_sha256"]:
        raise RuntimeError("Pinned source configuration changed")


def disarmed(plan: dict) -> bool:
    return (Path(plan["output"]) / "disarmed.json").exists()


def check_disarmed(plan: dict) -> None:
    if disarmed(plan):
        raise InterruptedError("Both follow-up watchdogs were cancelled by the user")


def wait_predecessor_stopped(plan: dict, role: str) -> None:
    source = json.loads(Path(plan["wm_plan"]).read_text())
    while any(
        process_alive(p)
        for p in source["tracked"]
        if p["role"] in {"launcher", "learner"}
    ):
        check_disarmed(plan)
        time.sleep(2)
    state(plan, role, "cleaning_completed_predecessor")
    release_predecessor(source)


def tagged_processes(plan: dict, role: str, proc_root: Path = Path("/proc")):
    expected = {
        os.fsencode("SPATIAL_FOLLOWUP_CHAIN=" + plan["output"]),
        os.fsencode("SPATIAL_FOLLOWUP_ROLE=" + role + "_workload"),
    }
    selected = {}
    for pid, process in snapshot().items():
        try:
            entries = set((proc_root / str(pid) / "environ").read_bytes().split(b"\0"))
        except OSError:
            continue
        if expected <= entries:
            selected[pid] = process
    return selected


def cleanup_workload(plan: dict, role: str) -> None:
    # Exact run tags AND same UID/start time are required for every signal.
    for signum, duration in ((signal.SIGTERM, 20), (signal.SIGKILL, 5)):
        targets = tagged_processes(plan, role)
        if not targets:
            return
        for process in targets.values():
            signal_process(process, signum)
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            if not tagged_processes(plan, role):
                return
            time.sleep(0.5)
    if tagged_processes(plan, role):
        raise RuntimeError(f"Some {role} workload processes did not stop")


def wait_idle(plan: dict, role: str) -> None:
    from queue_spatial_wm_update import gpu_usage

    last_report = float("-inf")
    while True:
        check_disarmed(plan)
        usage = gpu_usage()
        busy = {
            g: usage.get(g) for g in plan["gpus"] if g not in usage or usage[g] > 1024
        }
        if not busy:
            return
        if time.monotonic() - last_report >= 30:
            state(plan, role, "waiting_for_gpus", busy_mib=busy)
            last_report = time.monotonic()
        time.sleep(2)


def resume_candidates(plan: dict) -> list[Path]:
    candidates = []
    initial = Path(plan["initial_resume"])
    if checkpoint_complete(initial, 8):
        candidates.append(initial)
    for root in (Path(plan["source_run"]), Path(plan["output"]) / "grpo_primary"):
        logged = completed_step(root)
        for checkpoint in root.glob("*/checkpoints/global_step_*"):
            try:
                step = int(checkpoint.name.removeprefix("global_step_"))
            except ValueError:
                continue
            # A metadata file appearing during a still-failing save is not
            # sufficient: this optimizer step must also have been logged.
            if step <= logged and checkpoint_complete(checkpoint, 8):
                candidates.append(checkpoint)
    return candidates


def latest_resume(plan: dict) -> Path:
    choices = resume_candidates(plan)
    if not choices:
        raise RuntimeError(
            "No complete eight-rank policy/optimizer checkpoint is available"
        )
    return max(
        choices,
        key=lambda p: (int(p.name.removeprefix("global_step_")), p.stat().st_mtime_ns),
    )


def create_wan_deployment(plan: dict, pair: dict) -> Path:
    output = Path(plan["output"]) / "updated_wan_deployment"
    output.mkdir(exist_ok=False)
    sources = {
        "model-00001.safetensors": Path(pair["wan"]) / "model-00001.safetensors",
        "Wan2.2_VAE.pth": Path(pair["wan"]) / "Wan2.2_VAE.pth",
        "resnet_rm.pth": Path(plan["baseline_wan"]) / "resnet_rm.pth",
        "dataset": Path(plan["baseline_wan"]) / "dataset",
    }
    for name, source in sources.items():
        if source.exists():
            (output / name).symlink_to(
                source.resolve(), target_is_directory=source.is_dir()
            )
        elif name != "dataset":
            raise FileNotFoundError(source)
    return output


def grpo_command(
    plan: dict, role: str, checkpoint: Path, pair_path: Path | None = None
):
    start = int(checkpoint.name.removeprefix("global_step_"))
    target = start + plan["additional_grpo_steps"]
    output = Path(plan["output"]) / f"grpo_{role}"
    saves = [s for s in range(start + 1, target + 1) if s % 5 == 0 or s == target]
    env = dict(os.environ)
    for key in (
        "SPATIAL_WOVR_MODEL_PAIR",
        "SPATIAL_WOVR_BASELINE_WAN",
        "SPATIAL_WOVR_BASELINE_COSMOS",
    ):
        env.pop(key, None)
    env.update(
        SPATIAL_WOVR_OUTPUT=str(output),
        SPATIAL_WOVR_GPUS=",".join(plan["gpus"]),
        SPATIAL_WOVR_PLACEMENT="0-7",
        SPATIAL_WOVR_MONITOR_MEMORY="1",
        SPATIAL_WOVR_RUNTIME=str(Path(plan["output"]) / f"runtime_{role}"),
        SPATIAL_WOVR_TMP_ROOT=str(PROJECT.parent / "r"),
        SPATIAL_WOVR_RESOLVED_TEMPLATE=plan["source_config"],
        SPATIAL_WOVR_LAUNCHER=str(PROJECT / "experiments/spatial_wovr_pair_launch.py"),
        SPATIAL_COSMOS_PORT=str(18159 if role == "primary" else 18259),
        WAN_SPATIAL_CHECKPOINT=plan["baseline_wan"],
        COSMOS_WRIST_CHECKPOINT=plan["baseline_cosmos"],
        PI05_CHECKPOINT_ROOT=plan["pi05_checkpoint_root"],
    )
    if pair_path is not None:
        pair = json.loads(pair_path.read_text())
        env.update(
            SPATIAL_WOVR_MODEL_PAIR=str(pair_path),
            SPATIAL_WOVR_BASELINE_WAN=plan["baseline_wan"],
            SPATIAL_WOVR_BASELINE_COSMOS=plan["baseline_cosmos"],
            WAN_SPATIAL_CHECKPOINT=str(Path(plan["output"]) / "updated_wan_deployment"),
            COSMOS_WRIST_CHECKPOINT=pair["cosmos"],
        )
    command = [
        "bash",
        str(PROJECT / "experiments/run_spatial_wovr_shared_pool_8gpu.sh"),
        f"runner.resume_dir={checkpoint}",
        f"runner.max_steps={target}",
        f"runner.max_epochs={target}",
        "runner.save_interval=5",
        "runner.val_check_interval=5",
        f"actor.fsdp_config.save_training_state_at_steps={json.dumps(saves, separators=(',', ':'))}",
        f"runner.logger.experiment_name=spatial_followup_{role}",
    ]
    return command, env, output, target


def run_child(
    plan: dict, role: str, stage: str, command: list[str], environment: dict, **details
) -> None:
    check_disarmed(plan)
    log = Path(plan["output"]) / "logs" / f"{role}_{stage}.log"
    env = dict(
        environment,
        SPATIAL_FOLLOWUP_CHAIN=plan["output"],
        SPATIAL_FOLLOWUP_ROLE=role + "_workload",
        PYTHONUNBUFFERED="1",
    )
    with log.open("a") as stream:
        child = subprocess.Popen(
            command,
            cwd=PROJECT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        identity = read_process(child.pid)
        state(
            plan,
            role,
            "running",
            stage=stage,
            log=str(log),
            child_pid=child.pid,
            child_start_ticks=identity.started if identity else None,
            **details,
        )
        mail_started = False
        try:
            while child.poll() is None:
                check_disarmed(plan)
                if stage == "grpo" and not mail_started:
                    run = Path(details["grpo_output"])
                    if (run / "train.log").exists() and (
                        run / "resolved_config.yaml"
                    ).exists():
                        mail_env = dict(
                            env,
                            SPATIAL_WATCH_RUN=str(run),
                            SPATIAL_WATCH_LOG_ROOT=str(
                                Path(plan["output"]) / "email_watchdogs"
                            ),
                            SPATIAL_FOLLOWUP_ROLE=role + "_notification",
                        )
                        name = f"followup-mail-{role}-{Path(plan['output']).name[-10:]}"
                        entry = {
                            "role": role,
                            "screen": name,
                            "run": str(run),
                            "utc": utc(),
                        }
                        try:
                            notification = subprocess.run(
                                [
                                    "screen",
                                    "-dmS",
                                    name,
                                    "bash",
                                    str(
                                        PROJECT
                                        / "experiments/run_spatial_wovr_watchdog.sh"
                                    ),
                                ],
                                env=mail_env,
                                text=True,
                                capture_output=True,
                                timeout=15,
                                check=False,
                            )
                            entry["returncode"] = notification.returncode
                        except (OSError, subprocess.SubprocessError) as exc:
                            entry["error"] = type(exc).__name__
                        mail_started = True
                        try:
                            with (
                                Path(plan["output"]) / "notification_launches.jsonl"
                            ).open("a") as monitor:
                                monitor.write(json.dumps(entry) + "\n")
                        except OSError:
                            print(
                                f"Notification startup could not be recorded: {entry}",
                                flush=True,
                            )
                time.sleep(2)
        except BaseException:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            raise
        if child.returncode:
            raise RuntimeError(f"{stage} exited {child.returncode}: {log}")
    state(plan, role, "stage_complete", stage=stage, **details)


def finished_grpo(output: Path, target: int) -> bool:
    exit_path = output / "exit_code"
    if (
        not exit_path.exists()
        or exit_path.read_text().strip() != "0"
        or completed_step(output) < target
    ):
        return False
    return any(
        checkpoint_complete(p, 8)
        for p in output.glob(f"*/checkpoints/global_step_{target}")
    )


def run_grpo(
    plan: dict, role: str, checkpoint: Path, pair_path: Path | None = None
) -> None:
    verify(plan)
    if not checkpoint_complete(checkpoint, 8):
        raise RuntimeError(f"Incomplete policy/optimizer resume: {checkpoint}")
    command, env, output, target = grpo_command(plan, role, checkpoint, pair_path)
    wait_idle(plan, role)
    run_child(
        plan,
        role,
        "grpo",
        command,
        env,
        resume=str(checkpoint),
        target_step=target,
        grpo_output=str(output),
        updated_world_model=pair_path is not None,
    )
    if not finished_grpo(output, target):
        raise RuntimeError(
            f"GRPO exited without complete step-{target} optimizer state"
        )


def workload_lock(plan: dict, role: str):
    stream = (Path(plan["output"]) / "workload.lock").open("a")
    while True:
        check_disarmed(plan)
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return stream
        except BlockingIOError:
            time.sleep(1)


def primary(plan: dict) -> None:
    output = Path(plan["output"])
    process = read_process(os.getpid())
    write_json(
        output / "primary_identity.json",
        {"pid": process.pid, "start_ticks": process.started},
    )
    lock = None
    try:
        verify(plan)
        lock = workload_lock(plan, "primary")
        state(plan, "primary", "waiting_for_grpo", target_step=20)
        run_child(
            plan,
            "primary",
            "world_model_update",
            [
                plan["python"],
                str(PROJECT / "experiments/queue_spatial_wm_update.py"),
                "run",
                "--plan",
                plan["wm_plan"],
            ],
            dict(os.environ, CUDA_VISIBLE_DEVICES=""),
        )
        pair_path = Path(plan["pair_manifest"])
        pair = json.loads(pair_path.read_text())
        create_wan_deployment(plan, pair)
        source = json.loads(Path(plan["wm_plan"]).read_text())
        run_grpo(plan, "primary", Path(source["policy_checkpoint"]), pair_path)
        cleanup_workload(plan, "primary")
        write_json(
            output / "primary_done.json",
            {"completed_utc": utc(), "pair": str(pair_path)},
        )
        state(plan, "primary", "complete")
    except (Exception, KeyboardInterrupt) as exc:
        state(
            plan,
            "primary",
            "cleaning_failed_workload",
            error=f"{type(exc).__name__}: {exc}",
        )
        try:
            cleanup_workload(plan, "primary")
        finally:
            write_json(
                output / "primary_failed.json",
                {"failed_utc": utc(), "error": f"{type(exc).__name__}: {exc}"},
            )
        state(
            plan, "primary", "cancelled" if disarmed(plan) else "failed", error=str(exc)
        )
        raise
    finally:
        if lock is not None:
            lock.close()


def primary_outcome(plan: dict) -> str:
    output = Path(plan["output"])
    if disarmed(plan):
        return "cancelled"
    if (output / "primary_done.json").exists():
        return "success"
    if (output / "primary_failed.json").exists():
        return "failure"
    identity = output / "primary_identity.json"
    if identity.exists() and not process_alive(json.loads(identity.read_text())):
        # Recover a completed job even if only its supervisor died before it
        # could publish the final success marker.
        p = output / "primary_state.json"
        details = json.loads(p.read_text()) if p.exists() else {}
        if details.get("target_step") and finished_grpo(
            output / "grpo_primary", details["target_step"]
        ):
            return "success"
        return "failure"
    return "pending"


def fallback(plan: dict) -> None:
    output = Path(plan["output"])
    process = read_process(os.getpid())
    write_json(
        output / "fallback_identity.json",
        {"pid": process.pid, "start_ticks": process.started},
    )
    state(plan, "fallback", "standby")
    lock = None
    try:
        verify(plan)
        while (outcome := primary_outcome(plan)) == "pending":
            time.sleep(2)
        if outcome != "failure":
            state(
                plan,
                "fallback",
                "cancelled" if outcome == "cancelled" else "not_needed",
            )
            return
        lock = workload_lock(plan, "fallback")
        # Handles SIGKILL of the first watchdog: its tagged children may still
        # be alive even though its advisory workload lock has been released.
        cleanup_workload(plan, "primary")
        wait_predecessor_stopped(plan, "fallback")
        checkpoint = latest_resume(plan)
        state(
            plan,
            "fallback",
            "selected_resume",
            checkpoint=str(checkpoint),
            additional_steps=plan["additional_grpo_steps"],
            original_world_models=True,
        )
        run_grpo(plan, "fallback", checkpoint)
        cleanup_workload(plan, "fallback")
        write_json(
            output / "fallback_done.json",
            {"completed_utc": utc(), "resume": str(checkpoint)},
        )
        state(plan, "fallback", "complete")
    except (Exception, KeyboardInterrupt) as exc:
        cleanup_workload(plan, "fallback")
        state(
            plan,
            "fallback",
            "cancelled" if disarmed(plan) else "failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    finally:
        if lock is not None:
            lock.close()


def arm(args) -> None:
    import yaml

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new chain directory: {output}")
    output.mkdir(parents=True)
    (output / "logs").mkdir()
    wm_output = output / "world_model_update"
    arm_wm_update(
        SimpleNamespace(
            run_dir=args.run_dir,
            output=wm_output,
            watchdog_state=args.watchdog_state,
            gpus="0,1,2,3,4,5,6,7",
            target_step=20,
            per_task=50,
            steps=2000,
            save_interval=200,
            seed=20261021,
            settings=PROJECT / "experiments/config/spatial_wm_update.json",
        )
    )
    wm_plan = json.loads((wm_output / "plan.json").read_text())
    config = wm_output / "inputs/policy_config.yaml"
    cfg = yaml.safe_load(config.read_text())
    metadata = json.loads((args.run_dir / "cosmos_rank0_metadata.json").read_text())
    plan = {
        "schema": "spatial-ordered-watchdogs-v1",
        "created_utc": utc(),
        "output": str(output),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "source_run": str(args.run_dir.resolve()),
        "source_config": str(config),
        "source_config_sha256": sha256(config),
        "wm_plan": str(wm_output / "plan.json"),
        "pair_manifest": str(wm_output / "pair_2000/pair.json"),
        "python": sys.executable,
        "gpus": list(map(str, range(8))),
        "additional_grpo_steps": 20,
        "initial_resume": cfg["runner"]["resume_dir"],
        "baseline_wan": cfg["env"]["train"]["wan_wm_hf_ckpt_path"],
        "baseline_cosmos": metadata["checkpoint_identity"]["resolved_path"],
        "pi05_checkpoint_root": cfg["pi05_checkpoint_root"],
        "control_code": {
            name: sha256(PROJECT / "experiments" / name) for name in CONTROL_FILES
        },
    }
    if not checkpoint_complete(Path(plan["initial_resume"]), 8):
        raise RuntimeError("Original complete resume checkpoint is unavailable")
    if wm_plan["model_allocation"] != {
        "wan": ["0", "1", "2", "3"],
        "cosmos": ["4", "5", "6", "7"],
    }:
        raise ValueError("Unexpected world-model allocation")
    write_json(output / "plan.json", plan)
    for role in ROLES:
        state(plan, role, "prepared")


def cancel(plan: dict) -> None:
    if Path("/proc/sys/kernel/random/boot_id").read_text().strip() != plan["boot_id"]:
        raise RuntimeError("Cancellation must execute on this chain's recorded node")
    output = Path(plan["output"])
    write_json(output / "disarmed.json", {"cancelled_utc": utc(), "by": "user"})
    for role in ROLES:
        identity = output / f"{role}_identity.json"
        if identity.exists():
            saved = json.loads(identity.read_text())
            p = read_process(saved["pid"])
            if p is not None and p.started == saved["start_ticks"]:
                signal_process(p, signal.SIGTERM)
        cleanup_workload(plan, role)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["arm", "primary", "fallback", "status", "cancel"]
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--watchdog-state", type=Path)
    parser.add_argument("--plan", type=Path)
    args = parser.parse_args()
    if args.command == "arm":
        if not args.output or not args.run_dir or not args.watchdog_state:
            parser.error("arm requires --output, --run-dir and --watchdog-state")
        arm(args)
        return 0
    if not args.plan:
        parser.error("A recorded --plan is required")
    plan = json.loads(args.plan.read_text())
    if args.command == "cancel":
        cancel(plan)
    elif args.command == "status":
        for role in ROLES:
            print((Path(plan["output"]) / f"{role}_state.json").read_text())
    else:

        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"Signal {signum}")

        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, interrupted)
        with (Path(plan["output"]) / f"{args.command}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (Path(plan["output"]) / f"{args.command}_identity.json").exists():
                raise RuntimeError(
                    "This watchdog already ran; choose a fresh chain rather than replaying it"
                )
            (primary if args.command == "primary" else fallback)(plan)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyboardInterrupt) as exc:
        raise SystemExit(str(exc)) from exc
