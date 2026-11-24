#!/usr/bin/env python3
"""Wait for one successful GRPO run, then collect and update both world models."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
CODE_FILES = [
    "wm_update.py",
    "wm_update_collect.py",
    "wm_update_data.py",
    "wm_update_wan.py",
    "wm_update_cosmos.py",
    "cosmos_recipe_data.py",
    "cosmos_recipe_train.py",
    "wm_update_memory.py",
    "wm_update_accept.py",
    "queue_spatial_wm_update.py",
    "run_spatial_wm_update_4gpu.sh",
    "run_spatial_wm_update_8gpu.sh",
]


def utc() -> str:
    return datetime.now(UTC).isoformat()


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def record(output: Path, status: str, **details) -> None:
    value = {"status": status, "updated_utc": utc(), "pid": os.getpid(), **details}
    write_json(output / "state.json", value)
    with (output / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
    print(json.dumps(value, sort_keys=True), flush=True)


def completed_step(run: Path) -> int:
    path = run / "metrics.log"
    if not path.exists():
        return 0
    matches = re.findall(r"Global Step:\s*(\d+)\s*/", path.read_text(errors="replace"))
    return max(map(int, matches), default=0)


def process_alive(identity: dict) -> bool:
    from stop_spatial_grpo import read_process

    current = read_process(identity["pid"])
    return bool(
        current
        and current.uid == os.geteuid()
        and current.started == identity["start_ticks"]
    )


def predecessor_ready(plan: dict) -> bool:
    """A checkpoint alone never authorizes starting the next GPU workload."""
    run = Path(plan["grpo_run"])
    if sha256(run / "resolved_config.yaml") != plan["source_config_sha256"]:
        raise RuntimeError("The pinned GRPO configuration changed")
    exit_path = run / "exit_code"
    if not exit_path.exists():
        return False
    code = int(exit_path.read_text().strip())
    if code != 0:
        raise RuntimeError(f"GRPO exited with code {code}; no collection was started")
    if completed_step(run) < plan["target_step"]:
        raise RuntimeError("GRPO exited before the requested final step")
    weights = Path(plan["policy_checkpoint"]) / "actor/model_state_dict/full_weights.pt"
    if not weights.is_file() or weights.stat().st_size == 0:
        raise RuntimeError("Successful GRPO exit has no complete final policy weights")
    # exit_code is written just before the launcher's own finally/cleanup.
    return not any(
        process_alive(p)
        for p in plan["tracked"]
        if p["role"] in {"launcher", "learner"}
    )


def owned_remaining(plan: dict):
    """Only the same predecessor identities and their live descendants qualify."""
    from stop_spatial_grpo import protected_ancestors, snapshot

    processes = snapshot()
    excluded = protected_ancestors(processes, os.getpid())
    selected = {
        p["pid"]: processes[p["pid"]]
        for p in plan["release_processes"]
        if p["pid"] in processes
        and p["pid"] not in excluded
        and processes[p["pid"]].started == p["start_ticks"]
    }
    while True:
        children = {
            pid: p
            for pid, p in processes.items()
            if pid not in selected and pid not in excluded and p.ppid in selected
        }
        if not children:
            return selected
        selected.update(children)


def release_predecessor(plan: dict) -> None:
    """Reap this completed run's residual Ray workers, with PID-reuse checks."""
    from stop_spatial_grpo import signal_process

    for signum, seconds in ((signal.SIGTERM, 15), (signal.SIGKILL, 5)):
        targets = owned_remaining(plan)
        if not targets:
            return
        for process in targets.values():
            if signal_process(process, signum):
                print(
                    f"Completed GRPO cleanup: {signum.name} PID={process.pid}",
                    flush=True,
                )
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline and owned_remaining(plan):
            time.sleep(0.5)
    if owned_remaining(plan):
        raise RuntimeError(
            "Some completed GRPO workers did not release; no new GPU job was started"
        )


def gpu_usage() -> dict[str, int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=True,
        timeout=20,
    )
    return {
        row.split(",")[0].strip(): int(row.split(",")[1])
        for row in result.stdout.splitlines()
        if row.strip()
    }


def gpu_compute_pids(devices: list[str]) -> dict[str, list[int]]:
    """A low-memory CUDA process is still an active job, even below 1 GiB."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        text=True,
        capture_output=True,
        check=True,
        timeout=20,
    )
    mapping = {
        row.split(",")[1].strip(): row.split(",")[0].strip()
        for row in result.stdout.splitlines()
        if row.strip()
    }
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        text=True,
        capture_output=True,
        check=True,
        timeout=20,
    )
    active = {}
    for row in result.stdout.splitlines():
        fields = [v.strip() for v in row.split(",")]
        if len(fields) != 2 or not fields[1].isdecimal():
            continue
        device = mapping.get(fields[0])
        if device in devices:
            active.setdefault(device, []).append(int(fields[1]))
    return active


def stages_for(plan: dict) -> list[dict]:
    output = Path(plan["output"])
    base = [plan["python"], str(PROJECT / "experiments/wm_update.py")]
    devices = ",".join(plan["gpus"])
    policy = [
        "--policy-config",
        str(output / "inputs/policy_config.yaml"),
        "--policy-checkpoint",
        plan["policy_checkpoint"],
        "--mode",
        "train",
        "--max-steps",
        "240",
        "--settle-steps",
        "10",
        "--seed",
        str(plan["seed"]),
    ]
    train_options = [
        "--gpus",
        devices,
        "--model",
        "both",
        "--ranks-per-model",
        "4",
        "--accumulation",
        "4",
        "--lr",
        "1e-6",
        "--settings",
        str(output / "inputs/settings.json"),
        "--seed",
        str(plan["seed"]),
    ]
    stages = []

    def add(name: str, command: str, options: list[str]):
        stages.append(
            {
                "name": name,
                "command": base + [command] + options,
                "log": str(output / "logs" / f"{name}.log"),
            }
        )

    add(
        "pilot_collect",
        "collect",
        ["--gpus", devices, "--per-task", "2", "--output", str(output / "pilot")]
        + policy,
    )
    add(
        "pilot_prepare",
        "prepare",
        [
            "--collection",
            str(output / "pilot"),
            "--output",
            str(output / "pilot_manifest.json"),
            "--verify-rgb",
        ],
    )
    add(
        "pilot_inspect",
        "inspect",
        ["--manifest", str(output / "pilot_manifest.json"), "--verify-files"],
    )
    add(
        "backward_probe",
        "probe",
        train_options
        + [
            "--manifest",
            str(output / "pilot_manifest.json"),
            "--output",
            str(output / "probe"),
        ],
    )
    add(
        "collect",
        "collect",
        [
            "--gpus",
            devices,
            "--per-task",
            str(plan["per_task"]),
            "--output",
            str(output / "collection"),
        ]
        + policy,
    )
    add(
        "prepare",
        "prepare",
        [
            "--collection",
            str(output / "collection"),
            "--output",
            str(output / "manifest.json"),
            "--verify-rgb",
        ],
    )
    add(
        "inspect",
        "inspect",
        ["--manifest", str(output / "manifest.json"), "--verify-files"],
    )
    add(
        "train",
        "train",
        train_options
        + [
            "--manifest",
            str(output / "manifest.json"),
            "--output",
            str(output / "train"),
            "--steps",
            str(plan["steps"]),
            "--save-interval",
            str(plan["save_interval"]),
        ],
    )
    add(
        "package",
        "package",
        [
            "--manifest",
            str(output / "manifest.json"),
            "--wan-checkpoint",
            str(output / "train/wan" / f"step_{plan['steps']:06d}"),
            "--cosmos-checkpoint",
            str(
                output
                / "train/cosmos/spatial_wmevo/cosmos_wrist_v3/train/checkpoints"
                / f"iter_{plan['steps']:09d}"
            ),
            "--output",
            str(output / f"pair_{plan['steps']}"),
        ],
    )
    return stages


def arm(args) -> None:
    from stop_spatial_grpo import snapshot
    from wm_update import gpu_pool

    run, output = args.run_dir.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a fresh queue output: {output}")
    watchdog = json.loads(args.watchdog_state.read_text())
    if Path(watchdog["run_dir"]).resolve() != run:
        raise ValueError("Watchdog tracks a different GRPO run")
    processes = snapshot()
    tracked = watchdog["tracked"]
    if not all(
        process_alive(p) for p in tracked if p["role"] in {"launcher", "learner"}
    ):
        raise RuntimeError(
            "Expected GRPO launcher and learner must still be running when armed"
        )
    roots = {
        p["pid"]
        for p in tracked
        if p["role"] in {"learner", "cosmos"}
        and p["pid"] in processes
        and processes[p["pid"]].started == p["start_ticks"]
    }
    selected = set(roots)
    while True:
        children = {
            pid for pid, p in processes.items() if p.ppid in selected
        } - selected
        if not children:
            break
        selected.update(children)
    devices = gpu_pool(args.gpus)
    if len(devices) != 8:
        raise ValueError("This continuation uses exactly eight GPUs")
    config = run / "resolved_config.yaml"
    import yaml

    cfg = yaml.safe_load(config.read_text())
    if (
        cfg["runner"]["max_steps"] != args.target_step
        or cfg["runner"]["max_epochs"] != args.target_step
    ):
        raise ValueError("Queue target must match the pinned GRPO final step")
    if cfg["env"]["eval"]["task_suite_name"] != "libero_spatial":
        raise ValueError("Queue requires LIBERO Spatial")
    checkpoint = (
        run
        / cfg["runner"]["logger"]["experiment_name"]
        / "checkpoints"
        / f"global_step_{args.target_step}"
    )
    output.mkdir(parents=True)
    (output / "inputs").mkdir()
    (output / "logs").mkdir()
    shutil.copyfile(config, output / "inputs/policy_config.yaml")
    shutil.copyfile(args.settings, output / "inputs/settings.json")
    plan = {
        "schema": "spatial-grpo-to-wm-v1",
        "created_utc": utc(),
        "grpo_run": str(run),
        "source_config_sha256": sha256(config),
        "output": str(output),
        "policy_checkpoint": str(checkpoint),
        "target_step": args.target_step,
        "python": sys.executable,
        "gpus": devices,
        "tracked": tracked,
        "release_processes": [
            {"pid": pid, "start_ticks": processes[pid].started}
            for pid in sorted(selected)
        ],
        "steps": args.steps,
        "save_interval": args.save_interval,
        "per_task": args.per_task,
        "seed": args.seed,
        "poll_seconds": 5,
        "wait_for_gpus": bool(getattr(args, "wait_for_gpus", False)),
        "min_free_gib": 128,
        "code": {name: sha256(PROJECT / "experiments" / name) for name in CODE_FILES},
        "input_hashes": {
            name: sha256(output / "inputs" / name)
            for name in ["policy_config.yaml", "settings.json"]
        },
        "model_allocation": {"wan": devices[:4], "cosmos": devices[4:]},
        "accumulation": 4,
        "effective_batch_per_model": 16,
    }
    plan["stages"] = stages_for(plan)
    write_json(output / "plan.json", plan)
    record(
        output,
        "prepared",
        completed_grpo_step=completed_step(run),
        target_step=args.target_step,
        plan=str(output / "plan.json"),
    )


def verify_plan(plan: dict) -> None:
    if plan["code"] != {
        name: sha256(PROJECT / "experiments" / name) for name in CODE_FILES
    }:
        raise RuntimeError(
            "Queued update code changed; review and re-arm instead of silently changing the run"
        )
    output = Path(plan["output"])
    for name, expected in plan["input_hashes"].items():
        if sha256(output / "inputs" / name) != expected:
            raise RuntimeError(f"Queued input changed: {name}")
    if plan["stages"] != stages_for(plan):
        raise RuntimeError("Queued commands differ from the recorded configuration")


def start_saved_policy(args) -> Path:
    """Prepare an immediate run from an explicitly selected existing policy."""
    from wm_update import gpu_pool

    output = args.output.resolve()
    config = args.policy_config.resolve()
    checkpoint = args.policy_checkpoint.resolve()
    weights = checkpoint / "actor/model_state_dict/full_weights.pt"
    if not weights.is_file() or weights.stat().st_size == 0:
        raise ValueError("An existing complete saved policy is required")
    if checkpoint.name != f"global_step_{args.policy_step}":
        raise ValueError("Selected policy step does not match the checkpoint directory")
    if output.exists():
        raise FileExistsError(f"Choose a fresh update output: {output}")
    devices = gpu_pool(args.gpus)
    if len(devices) != 8:
        raise ValueError("This continuation uses exactly eight GPUs")
    node = subprocess.run(
        ["hostname", "-I"], check=True, capture_output=True, text=True, timeout=10
    )
    if args.expected_node_ip not in node.stdout.split():
        raise RuntimeError(f"Refusing to start outside node {args.expected_node_ip}")
    import yaml

    cfg = yaml.safe_load(config.read_text())
    if cfg["env"]["eval"]["task_suite_name"] != "libero_spatial":
        raise ValueError("Collection requires LIBERO Spatial")
    output.mkdir(parents=True)
    (output / "inputs").mkdir()
    (output / "logs").mkdir()
    shutil.copyfile(config, output / "inputs/policy_config.yaml")
    shutil.copyfile(args.settings, output / "inputs/settings.json")
    stat = weights.stat()
    plan = {
        "schema": "spatial-grpo-to-wm-v1",
        "created_utc": utc(),
        "startup_mode": "saved_policy",
        "expected_node_ip": args.expected_node_ip,
        "execution_boot_id": Path("/proc/sys/kernel/random/boot_id")
        .read_text()
        .strip(),
        "grpo_run": str(config.parent),
        "source_config_sha256": sha256(config),
        "source_config_path": str(config),
        "output": str(output),
        "policy_checkpoint": str(checkpoint),
        "target_step": args.policy_step,
        "policy_weight_identity": {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": sha256(weights),
        },
        "python": sys.executable,
        "gpus": devices,
        "tracked": [],
        "release_processes": [],
        "steps": args.steps,
        "save_interval": args.save_interval,
        "per_task": args.per_task,
        "seed": args.seed,
        "poll_seconds": 5,
        "wait_for_gpus": bool(getattr(args, "wait_for_gpus", False)),
        "min_free_gib": 128,
        "code": {name: sha256(PROJECT / "experiments" / name) for name in CODE_FILES},
        "input_hashes": {
            name: sha256(output / "inputs" / name)
            for name in ["policy_config.yaml", "settings.json"]
        },
        "model_allocation": {"wan": devices[:4], "cosmos": devices[4:]},
        "accumulation": 4,
        "effective_batch_per_model": 16,
    }
    plan["stages"] = stages_for(plan)
    write_json(output / "plan.json", plan)
    record(
        output,
        "prepared",
        policy_step=args.policy_step,
        node=args.expected_node_ip,
        plan=str(output / "plan.json"),
    )
    return output / "plan.json"


def verify_saved_policy(plan: dict) -> None:
    """Check the actual node and saved weights without consulting remote PIDs."""
    node = subprocess.run(
        ["hostname", "-I"], check=True, capture_output=True, text=True, timeout=10
    )
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if (
        plan["expected_node_ip"] not in node.stdout.split()
        or boot != plan["execution_boot_id"]
    ):
        raise RuntimeError("Saved-policy plan belongs to a different execution node")
    if sha256(Path(plan["source_config_path"])) != plan["source_config_sha256"]:
        raise RuntimeError("Selected policy configuration changed")
    weights = Path(plan["policy_checkpoint"]) / "actor/model_state_dict/full_weights.pt"
    stat = weights.stat()
    expected = plan["policy_weight_identity"]
    if (
        stat.st_size != expected["size"]
        or stat.st_mtime_ns != expected["mtime_ns"]
        or sha256(weights) != expected["sha256"]
    ):
        raise RuntimeError("Selected policy weights changed")


def wait_for_gpus(plan: dict, output: Path) -> None:
    """Wait only when requested; never signal any process occupying a GPU."""
    deadline = time.monotonic() + 120
    last_report = float("-inf")
    while True:
        used = gpu_usage()
        busy = {g: used.get(g) for g in plan["gpus"] if g not in used or used[g] > 1024}
        active = {}
        if not busy and plan.get("wait_for_gpus", False):
            active = gpu_compute_pids(plan["gpus"])
            busy = {gpu: used[gpu] for gpu in active}
        if not busy:
            return
        if plan.get("wait_for_gpus", False):
            now = time.monotonic()
            if now - last_report >= 30:
                record(
                    output,
                    "waiting_for_gpus",
                    busy_mib=busy,
                    node=plan.get("expected_node_ip"),
                    policy_step=plan["target_step"],
                    active_compute_pids=active,
                )
                last_report = now
            time.sleep(plan.get("poll_seconds", 5))
            continue
        if plan.get("startup_mode") == "saved_policy":
            raise RuntimeError(
                f"Requested GPUs are occupied: {busy}; no new job was started"
            )
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"GPUs remain occupied after completed GRPO cleanup: {busy}; other jobs were retained"
            )
        time.sleep(2)


def run_queue(plan_path: Path) -> None:
    plan = json.loads(plan_path.read_text())
    output = Path(plan["output"])
    lock = (output / "queue.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = json.loads((output / "state.json").read_text())
    if state["status"] not in {"prepared", "waiting_for_grpo"}:
        raise RuntimeError(
            "Queue was already started or failed; inspect state instead of replaying training"
        )
    child = None

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, interrupted)
    try:
        verify_plan(plan)
        saved_policy = plan.get("startup_mode") == "saved_policy"
        if saved_policy:
            verify_saved_policy(plan)
            record(
                output,
                "starting_saved_policy",
                policy_step=plan["target_step"],
                node=plan["expected_node_ip"],
            )
        else:
            record(
                output,
                "waiting_for_grpo",
                target_step=plan["target_step"],
                completed_grpo_step=completed_step(Path(plan["grpo_run"])),
            )
            missing_since = None
            while not predecessor_ready(plan):
                alive = any(
                    process_alive(p)
                    for p in plan["tracked"]
                    if p["role"] in {"launcher", "learner"}
                )
                if not alive:
                    missing_since = missing_since or time.monotonic()
                    if time.monotonic() - missing_since > 60:
                        raise RuntimeError(
                            "GRPO processes stopped without successful final completion"
                        )
                else:
                    missing_since = None
                time.sleep(plan["poll_seconds"])
            verify_plan(plan)
            record(output, "releasing_completed_grpo")
            release_predecessor(plan)
        wait_for_gpus(plan, output)
        if saved_policy and plan.get("wait_for_gpus"):
            # The wait may span hours; verify again before using the selected policy.
            verify_saved_policy(plan)
        # A failed later stage must not reload eight policies and recollect
        # already committed episodes. Only reuse stages with recorded success.
        events = [
            json.loads(line)
            for line in (output / "events.jsonl").read_text().splitlines()
            if line.strip()
        ]
        completed = {
            event["stage"]
            for event in events
            if event.get("status") == "stage_complete"
        }
        for stage in plan["stages"]:
            verify_plan(plan)
            if stage["name"] in completed:
                record(output, "stage_reused", stage=stage["name"])
                continue
            if plan.get("wait_for_gpus") and stage["name"] in {
                "pilot_collect",
                "backward_probe",
                "collect",
                "train",
            }:
                wait_for_gpus(plan, output)
            free = shutil.disk_usage(output).free / 2**30
            if free < plan["min_free_gib"]:
                raise RuntimeError(
                    f"Insufficient disk headroom before {stage['name']}: {free:.1f} GiB"
                )
            record(output, "running", stage=stage["name"], log=stage["log"])
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1")
            with Path(stage["log"]).open("a") as stream:
                child = subprocess.Popen(
                    stage["command"],
                    cwd=PROJECT,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                code = child.wait()
                child = None
            if code:
                raise RuntimeError(
                    f"Update stage {stage['name']} exited {code}: {stage['log']}"
                )
            record(output, "stage_complete", stage=stage["name"], log=stage["log"])
        record(output, "complete", pair=str(output / f"pair_{plan['steps']}"))
    except (Exception, KeyboardInterrupt) as exc:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        record(output, "failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        lock.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["arm", "start", "run", "status"])
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--watchdog-state", type=Path)
    parser.add_argument("--policy-config", type=Path)
    parser.add_argument("--policy-checkpoint", type=Path)
    parser.add_argument("--policy-step", type=int, default=15)
    parser.add_argument("--expected-node-ip")
    parser.add_argument(
        "--wait-for-gpus",
        action="store_true",
        help="Record an idle-GPU wait in a new start/arm plan; never evict other jobs",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--target-step", type=int, default=20)
    parser.add_argument("--per-task", type=int, default=50)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--save-interval", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument(
        "--settings",
        type=Path,
        default=PROJECT / "experiments/config/spatial_wm_update.json",
    )
    args = parser.parse_args()
    if args.wait_for_gpus and args.command not in {"start", "arm"}:
        parser.error("--wait-for-gpus belongs to start/arm; run uses its recorded plan")
    if min(args.target_step, args.per_task, args.steps, args.save_interval) < 1:
        parser.error("Steps, collection count and intervals must be positive")
    if args.command == "start":
        if (
            not args.policy_config
            or not args.policy_checkpoint
            or not args.output
            or not args.expected_node_ip
        ):
            parser.error(
                "start requires --policy-config, --policy-checkpoint, --output and --expected-node-ip"
            )
        run_queue(start_saved_policy(args))
    elif args.command == "arm":
        if not args.run_dir or not args.output or not args.watchdog_state:
            parser.error("arm requires --run-dir, --watchdog-state and --output")
        arm(args)
    else:
        if not args.plan:
            parser.error("run/status requires --plan")
        if args.command == "run":
            run_queue(args.plan)
        else:
            plan = json.loads(args.plan.read_text())
            state = json.loads((Path(plan["output"]) / "state.json").read_text())
            state["completed_grpo_step"] = completed_step(Path(plan["grpo_run"]))
            print(json.dumps(state, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
