"""Wait for GPUs, resume completed state, and recover from CUDA OOM exits."""

import argparse
import fcntl
import json
import os
import re
import signal
import socket
import subprocess
import time
from pathlib import Path

from spatial_wovr_memory_guard import (
    GuardSettings,
    MemoryGuard,
    atomic_json,
    checkpoint_complete,
)

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    PROJECT.parent
    / "grpo_runs/spatial_wovr20_fast_pi05_12000_8gpu/20261020_131300_node146/spatial_wovr20_wrist_v3_pi05_12000_8gpu_kir_first_close_pre16/checkpoints/global_step_10"
)


def checkpoint_step(path):
    match = re.fullmatch(r"global_step_(\d+)", Path(path).name)
    if match is None:
        raise ValueError("Resume directory must end with global_step_<number>")
    return int(match.group(1))


def latest_recovery(root, initial):
    selected, step = Path(initial), checkpoint_step(initial)
    for marker in Path(root).glob(
        "attempt_*/*/checkpoints/global_step_*/recovery_complete.json"
    ):
        try:
            data = json.loads(marker.read_text())
            current = checkpoint_step(marker.parent)
            valid = data["step"] == current and data["world_size"] == 8
            if valid and current > step and checkpoint_complete(marker.parent, 8):
                selected, step = marker.parent, current
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return selected


def build_launch(checkpoint, output, max_steps):
    env = dict(os.environ)
    env.update(
        SPATIAL_WOVR_OUTPUT=str(output),
        SPATIAL_WOVR_GPUS="0,1,2,3,4,5,6,7",
        SPATIAL_WOVR_PLACEMENT="0-7",
        SPATIAL_WOVR_FAST_TRANSFER="1",
        SPATIAL_WOVR_MONITOR_MEMORY="1",
        SPATIAL_WOVR_TMP_ROOT=str(PROJECT.parent / "r"),
        SPATIAL_WOVR_RUNTIME=str(
            PROJECT / "runtime_artifacts/spatial_wovr_guarded_8gpu"
        ),
    )
    command = [
        "bash",
        str(PROJECT / "experiments/run_spatial_wovr_20step.sh"),
        f"runner.resume_dir={Path(checkpoint)}",
        f"runner.max_steps={max_steps}",
        f"runner.max_epochs={max_steps}",
        "runner.memory_guard.enabled=true",
        "runner.memory_guard.checkpoint_each_step=true",
        "actor.memory_pool.target_fraction=0.95",
        "actor.memory_pool.device_headroom_gib=4.0",
        "env.train.cosmos_batch_size=4",
        "env.train.cosmos_wrist_parallel_generation=false",
        "env.train.fast_frame_transfer=true",
        "env.train.skip_successful_env_generation=true",
        "env.train.video_cfg.record_every_n_episodes=4",
    ]
    return command, env


def failed_with_oom(output, launcher_log=None):
    logs = [Path(output) / "train.log", *Path(output).glob("cosmos_rank*.log")]
    if launcher_log is not None:
        logs.append(Path(launcher_log))
    for log in logs:
        if not log.is_file():
            continue
        data = log.read_text(errors="replace").lower()
        if any(
            text in data
            for text in [
                "torch.outofmemoryerror: cuda out of memory",
                "repeated cuda out of memory without observable contention",
                "cuda error: out of memory",
            ]
        ):
            return True
    return False


def supervise(root, initial, max_steps):
    root.mkdir(parents=True, exist_ok=False)
    child = None

    def status(**fields):
        value = {"supervisor_pid": os.getpid(), "time": time.time(), **fields}
        atomic_json(root / "supervisor_state.json", value)
        print(json.dumps(value), flush=True)

    def interrupted(signum, frame):
        if child is not None and child.poll() is None:
            # This process group was created by this supervisor, never a
            # matching-name or device-wide kill.
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                status(status="stopping", child_pid=child.pid)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    attempt = 0
    while True:
        checkpoint = latest_recovery(root, initial)
        step = checkpoint_step(checkpoint)
        if step >= max_steps:
            status(status="complete", checkpoint=str(checkpoint), step=step)
            return 0
        attempt += 1
        output = root / f"attempt_{attempt:03d}_from_{step}"
        status(
            status="waiting_for_gpus",
            attempt=attempt,
            checkpoint=str(checkpoint),
            output=str(output),
        )
        settings = GuardSettings(enabled=True, poll_seconds=10, stable_checks=3)
        MemoryGuard(settings, list(range(8)), str(root)).wait(
            "startup",
            required_free_gib=77.0,
            force=True,
            require_stable=True,
        )
        command, env = build_launch(checkpoint, output, max_steps)
        log = root / f"attempt_{attempt:03d}_launcher.log"
        watch_started = False
        with log.open("w") as stream:
            child = subprocess.Popen(
                command,
                cwd=PROJECT,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            status(
                status="running",
                attempt=attempt,
                checkpoint=str(checkpoint),
                output=str(output),
                child_pid=child.pid,
                launcher_log=str(log),
            )
            while child.poll() is None:
                if not watch_started and (output / "train.log").is_file():
                    watch_name = f"spatial-guard-watch-{os.getpid()}-{attempt}"
                    try:
                        subprocess.run(
                            [
                                "screen",
                                "-dmS",
                                watch_name,
                                "env",
                                f"SPATIAL_WATCH_RUN={output}",
                                "bash",
                                str(
                                    PROJECT / "experiments/run_spatial_wovr_watchdog.sh"
                                ),
                            ],
                            cwd=PROJECT,
                            check=True,
                            timeout=15,
                        )
                        watch_started = True
                        status(
                            status="running",
                            attempt=attempt,
                            checkpoint=str(checkpoint),
                            output=str(output),
                            child_pid=child.pid,
                            watchdog=watch_name,
                        )
                    except (OSError, subprocess.SubprocessError) as exc:
                        print(
                            f"Watchdog startup retry: {type(exc).__name__}", flush=True
                        )
                time.sleep(5)
            code = child.returncode
        if code == 0:
            final = latest_recovery(root, initial)
            if checkpoint_step(final) < max_steps:
                status(
                    status="failed",
                    reason="Successful exit without final recovery checkpoint",
                    output=str(output),
                )
                return 1
            status(
                status="complete", checkpoint=str(final), step=checkpoint_step(final)
            )
            return 0
        if not failed_with_oom(output, log):
            status(
                status="failed",
                exit_code=code,
                output=str(output),
                reason="Non-OOM failure; inspect logs",
            )
            return code
        status(
            status="recovering",
            exit_code=code,
            output=str(output),
            checkpoint=str(latest_recovery(root, initial)),
            reason="CUDA OOM; wait for free GPUs and resume",
        )
        time.sleep(10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-dir", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--check-assets", action="store_true")
    args = parser.parse_args()
    initial = args.resume_dir.resolve()
    if not checkpoint_complete(initial, 8):
        parser.error(f"Incomplete eight-rank resume checkpoint: {initial}")
    if checkpoint_step(initial) >= args.max_steps:
        parser.error("The target step must be later than the resume checkpoint")
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime()) + f"_{os.getpid()}"
    root = (
        args.output_root
        or PROJECT.parent / "grpo_runs/spatial_wovr20_guarded_8gpu" / stamp
    ).resolve()
    if args.check_assets:
        command, env = build_launch(initial, root / "cpu_check", args.max_steps)
        return subprocess.run(
            command + ["--check-assets"], cwd=PROJECT, env=env, check=False
        ).returncode
    lock_root = PROJECT / "runtime_artifacts/training_queue"
    lock_root.mkdir(parents=True, exist_ok=True)
    host = re.sub(r"[^a-zA-Z0-9_.-]", "_", socket.gethostname())
    with (lock_root / f"guarded8_{host}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("A protected eight-GPU supervisor already runs on this host")
        return supervise(root, initial, args.max_steps)


if __name__ == "__main__":
    raise SystemExit(main())
