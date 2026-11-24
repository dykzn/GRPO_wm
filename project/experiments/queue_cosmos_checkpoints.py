"""Evaluate Cosmos checkpoints after Wan finishes or is explicitly stopped."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from queue_spatial_wm_update import (
    gpu_compute_pids, gpu_usage, process_alive, record, sha256, utc, write_json,
)
from spatial_followup_email import OutcomeMailer, read_json
from stop_spatial_grpo import read_process
from watch_training_email import DEFAULT_CONFIG

PROJECT = Path(__file__).resolve().parents[1]
CODE = (
    "queue_cosmos_checkpoints.py", "wm_eval.py", "wm_eval_data.py",
    "wm_eval_predict.py", "wm_eval_metrics.py", "spatial_wovr_launch.py",
    "spatial_wovr_runtime.py", "spatial_wovr_video.py",
    "queue_spatial_wm_update.py", "spatial_followup_email.py",
    "watch_training_email.py", "stop_spatial_grpo.py",
    "world_model_grpo/cosmos_wrist_adapter.py",
)
METRICS = ("FPS", "LPIPS", "FID", "FVD_variable_length", "FloLPIPS")
TERMINAL = {"complete", "failed", "cancelled"}


def identity():
    process = read_process(os.getpid())
    return {"pid": process.pid, "start_ticks": process.started}


def asset(path):
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def verify(plan):
    if Path("/proc/sys/kernel/random/boot_id").read_text().strip() != plan["boot_id"]:
        raise RuntimeError("This queue belongs to another execution node")
    for name, expected in plan["code"].items():
        if sha256(PROJECT / "experiments" / name) != expected:
            raise RuntimeError(f"Evaluation code changed after arming: {name}")
    for name, expected in plan["inputs"].items():
        if sha256(Path(name)) != expected:
            raise RuntimeError(f"Frozen queue input changed: {name}")
    for row in plan["assets"]:
        if asset(row["path"]) != row:
            raise RuntimeError(f"Evaluation asset changed: {row['path']}")


def inspect_checkpoint(path, step, manifest_sha256):
    complete = read_json(path / "wm_update_complete.json")
    if (complete.get("step") != step
            or complete.get("identity", {}).get("model") != "cosmos"
            or complete["identity"].get("manifest") != manifest_sha256):
        raise ValueError(f"Cosmos checkpoint has no matching completion record: {path}")
    model = path / "model"
    files = [model / ".metadata"] + [
        model / f"__{rank}_0.distcp" for rank in range(complete["world_size"])
    ]
    rows = [asset(file) for file in files]
    # Native Cosmos writes empty nonzero-rank shards; they are valid DCP files.
    if not rows[0]["size"] or not any(row["size"] for row in rows[1:]):
        raise ValueError(f"Cosmos checkpoint has no model state: {path}")
    return complete, rows


def arm(args):
    from wm_eval import configure_environment
    from wm_eval_metrics import HUB_WEIGHTS, dependencies

    configure_environment()
    upstream = read_json(args.training_plan)
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if upstream.get("schema") != "wan-local-eight-gpu-resume-v1" or upstream["boot_id"] != boot:
        raise ValueError("Select the current local-node Wan continuation plan")
    source = Path(upstream["output"])
    worker = read_json(source / "identity.json")
    if not worker:
        raise ValueError("The Wan workflow has no recorded process identity")
    steps = sorted(set(args.steps))
    if not steps or any(step <= 0 for step in steps) or len(steps) != len(args.steps):
        raise ValueError("Choose distinct positive Cosmos checkpoint steps")
    checkpoint_root = Path(upstream["cosmos_checkpoint"]).parent
    checkpoints, assets, inputs, training_identity = [], [], {}, None
    for step in steps:
        path = checkpoint_root / f"iter_{step:09d}"
        complete, files = inspect_checkpoint(path, step, upstream["training_manifest_sha256"])
        if training_identity is not None and complete["identity"] != training_identity:
            raise ValueError("Selected Cosmos checkpoints belong to different training runs")
        training_identity = complete["identity"]
        checkpoints.append({"step": step, "path": str(path.resolve())})
        assets.extend(files)
        inputs[str(path / "wm_update_complete.json")] = sha256(path / "wm_update_complete.json")
    reference_root = Path(upstream["cosmos_metrics_source"]) / "baseline"
    reference = read_json(reference_root / "cosmos_metrics.json")
    if reference.get("manifest_sha256") != upstream["evaluation_manifest_sha256"]:
        raise ValueError("Completed baseline metrics use another held-out split")
    runtimes = sorted((reference_root / "cosmos").glob("shard_*_runtime.json"))
    if not runtimes:
        raise ValueError("Baseline generation settings are unavailable")
    for path in runtimes:
        settings = read_json(path)["settings"]
        if any(settings.get(key) != value for key, value in {
            "manifest": upstream["evaluation_manifest_sha256"],
            "seed": upstream["seed"], "steps": 5, "warmup": 3, "batch": 1,
        }.items()):
            raise ValueError("Baseline inference settings differ from this comparison")
        inputs[str(path)] = sha256(path)
    input_paths = [args.training_plan, source / "inputs/eval_manifest.json",
                   reference_root / "cosmos_metrics.json"]
    immediate = getattr(args, "start_immediately", False)
    if immediate:
        input_paths.append(source / "stop_requested.json")
    inputs.update({str(path.resolve()): sha256(path) for path in input_paths})
    environment = {key: os.environ[key] for key in (
        "COSMOS_VAE", "COSMOS_DATASET_STATS", "COSMOS_T5_EMBEDDINGS",
        "COSMOS_BASE_ROOT", "COSMOS_PYTHON", "RLINF_ENV", "RLINF_BASE_ROOT",
        "LIBERO_ROOT", "LIBERO_BENCHMARK_ROOT", "OPENPI_ROOT", "DIFFSYNTH_ROOT",
        "WORLDPILOT_ROOT", "WM_METRIC_ASSETS", "TORCH_HOME",
    )}
    for key in ("COSMOS_VAE", "COSMOS_DATASET_STATS", "COSMOS_T5_EMBEDDINGS", "COSMOS_PYTHON"):
        assets.append(asset(environment[key]))
    expected_assets = {row["path"]: row for row in training_identity["assets"]}
    for key in ("COSMOS_VAE", "COSMOS_DATASET_STATS"):
        row = asset(environment[key])
        if expected_assets.get(row["path"]) != row:
            raise ValueError(f"Cosmos training and evaluation assets differ: {key}")
    dependencies_status = dependencies()
    if not dependencies_status["ready"]:
        raise ValueError(f"Missing metric dependencies: {dependencies_status['missing']}")
    metric_root = Path(environment["WM_METRIC_ASSETS"])
    assets.extend(asset(path) for path in [
        metric_root / "i3d_torchscript.pt", metric_root / "flolpips/flolpips.py",
        metric_root / "flolpips/pwcnet.py", metric_root / "flolpips/weights/v0.1/alex.pth",
        *(Path(environment["TORCH_HOME"]) / "hub/checkpoints" / name for name in HUB_WEIGHTS),
    ])
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "inputs").mkdir()
    (output / "logs").mkdir()
    manifest = output / "inputs/eval_manifest.json"
    shutil.copyfile(source / "inputs/eval_manifest.json", manifest)
    inputs[str(manifest)] = sha256(manifest)
    plan = {
        "schema": "cosmos-multi-checkpoint-after-wan-v1", "created_utc": utc(),
        "output": str(output), "boot_id": boot, "upstream": str(source),
        "upstream_identity": worker, "upstream_target": upstream["steps"],
        "upstream_training_manifest_sha256": upstream["training_manifest_sha256"],
        "python": upstream["python"], "gpus": upstream["gpus"], "seed": upstream["seed"],
        "checkpoints": checkpoints, "evaluation_manifest_sha256": upstream["evaluation_manifest_sha256"],
        "evaluation_summary": upstream["evaluation_summary"], "environment": environment,
        "baseline_metrics": str(reference_root / "cosmos_metrics.json"),
        "reference_fps_comparable": False, "reference_node": "previous evaluation node",
        "inference_steps": 5, "warmup_chunks": 3, "workers_per_gpu": 1,
        "allow_shared_gpus": getattr(args, "allow_shared_gpus", False),
        "minimum_free_mib": 32768,
        "gpu_total_mib": gpu_total_memory() if getattr(args, "allow_shared_gpus", False) else {},
        "inputs": inputs, "assets": assets,
        "code": {name: sha256(PROJECT / "experiments" / name) for name in CODE},
        "protocol_note": upstream["protocol_note"],
        "start_condition": "stopped_predecessor" if immediate else "completed_wan_workflow",
        "stop_request": str(source / "stop_requested.json") if immediate else None,
        "ordering": (
            "User stopped Wan: verify its recorded supervisor and all recorded descendants have exited, then evaluate Cosmos checkpoints sequentially on all eight GPUs."
            if immediate else
            "Wait for Wan training, then its existing video evaluation/cleanup, then Cosmos checkpoints sequentially on all eight GPUs. A completed Wan training remains eligible if its subsequent metric evaluation fails."
        ),
    }
    verify(plan)
    if immediate and not stopped_predecessor_ready(plan):
        raise RuntimeError("Wan or one of its recorded workers is still running")
    write_json(output / "plan.json", plan)
    record(output, "prepared", checkpoints=steps, next_stage="wait_wan_release" if immediate else "wait_wan")
    print(json.dumps({"plan": str(output / "plan.json"), "checkpoints": steps}, indent=2))


def stopped_predecessor_ready(plan):
    """An explicit local stop record may replace successful Wan completion."""
    request = read_json(Path(plan["stop_request"]))
    tracked = request.get("tracked", [])
    root = Path(plan["upstream"])
    if (request.get("request") != "user_requested_stop_wan_start_cosmos"
            or request.get("boot_id") != plan["boot_id"]
            or request.get("upstream_output") != str(root)
            or request.get("upstream_identity") != plan["upstream_identity"]
            or not any(all(row.get(key) == plan["upstream_identity"][key]
                           for key in ("pid", "start_ticks")) for row in tracked)):
        raise RuntimeError("Immediate evaluation needs a matching explicit Wan stop record")
    child = read_json(root / "active_child.json")
    if any(process_alive(row) for row in tracked) or (child and process_alive(child)):
        return False
    state = read_json(root / "state.json")
    if state.get("status") not in ("complete", "failed", "cancelled"):
        raise RuntimeError("Stopped Wan has no terminal workflow result")
    return True


def upstream_ready(plan):
    if plan.get("start_condition") == "stopped_predecessor":
        return stopped_predecessor_ready(plan)
    root = Path(plan["upstream"])
    state = read_json(root / "state.json")
    trained = read_json(root / "wan_training_complete.json")
    alive = process_alive(plan["upstream_identity"])
    if not alive:
        # Completion markers can be written between the state and process reads.
        state = read_json(root / "state.json")
        trained = read_json(root / "wan_training_complete.json")
    if trained.get("status") != "complete":
        if state.get("status") == "failed" or not alive:
            raise RuntimeError(f"Wan stopped before successful training: {state.get('error', state)}")
        return False
    if trained.get("step") != plan["upstream_target"]:
        raise RuntimeError("Wan completion marker has an unexpected final step")
    path = Path(trained["checkpoint"])
    complete = read_json(path / "complete.json")
    exported = read_json(path / "runtime/wm_update.json")
    if (complete.get("step") != plan["upstream_target"] or complete.get("world_size") != 8
            or complete.get("identity", {}).get("manifest") != plan["upstream_training_manifest_sha256"]
            or exported.get("step") != complete["step"]
            or exported.get("manifest_sha256") != plan["upstream_training_manifest_sha256"]
            or not (path / "state/.metadata").is_file()
            or not (path / "runtime/model-00001.safetensors").is_file()
            or not (path / "runtime/Wan2.2_VAE.pth").is_file()):
        raise RuntimeError("Wan final checkpoint or runtime export is incomplete")
    child = read_json(root / "active_child.json")
    if alive or (child and process_alive(child)):
        return False
    if state.get("status") not in ("complete", "failed"):
        raise RuntimeError("Wan supervisor exited without a terminal workflow result")
    return True


def package_checkpoint(root, checkpoint):
    source = Path(checkpoint["path"])
    package = root / "packages" / f"step_{checkpoint['step']:06d}"
    target = package / source.name
    target.mkdir(parents=True)
    (target / "model").symlink_to(source / "model", target_is_directory=True)
    files = [{"path": str(Path(source.name) / "model" / path.name),
              "size": path.stat().st_size, "sha256": sha256(path)}
             for path in sorted((source / "model").iterdir()) if path.is_file()]
    tree = hashlib.sha256("".join(f"{row['sha256']}  {row['path']}\n" for row in files).encode()).hexdigest()
    write_json(package / "checkpoint_manifest.json", {
        "checkpoint": source.name, "run_name": "spatial_wmevo",
        "format": "pytorch-distributed-checkpoint", "files": files,
        "file_count": len(files), "total_bytes": sum(row["size"] for row in files),
        "tree_sha256": tree,
    })
    return target


def evaluation_command(plan, destination):
    return [plan["python"], str(PROJECT / "experiments/wm_eval.py"), "run",
            "--model", "cosmos", "--manifest", str(destination / "eval_manifest.json"),
            "--output", str(destination), "--gpus", ",".join(plan["gpus"]),
            "--steps", str(plan["inference_steps"]), "--seed", str(plan["seed"]),
            "--warmup", str(plan["warmup_chunks"]), "--workers-per-gpu", "1"]


def write_summary(plan, *, final=False):
    root = Path(plan["output"])
    reference = read_json(Path(plan["baseline_metrics"]))
    rows, changes = [], {}
    for checkpoint in plan["checkpoints"]:
        step = checkpoint["step"]
        path = root / f"step_{step:06d}/cosmos_metrics.json"
        if not path.is_file() and not final:
            continue
        data = read_json(path)
        for key in ("model", "view", "manifest_sha256", "episodes", "frames", "protocol",
                    "metric_backbones", "metric_resolution", "native_resolution"):
            if data.get(key) != reference.get(key):
                raise ValueError(f"Checkpoint {step} comparison protocols differ: {key}")
        if any(not math.isfinite(data[key]) for key in METRICS) or data["FPS"] <= 0:
            raise ValueError(f"Checkpoint {step} has invalid metrics")
        rows.append({"step": step, "episodes": data["episodes"], "frames": data["frames"],
                     **{key: data[key] for key in METRICS}})
        changes[str(step)] = {key: data[key] - reference[key] for key in METRICS if key != "FPS"}
    result = {"rows": rows, "baseline_reference": reference,
              "changes_vs_baseline": changes, "reference_fps_comparable": False,
              "completed_checkpoints": len(rows), "requested_checkpoints": len(plan["checkpoints"]),
              "protocol_note": plan["protocol_note"],
              "fps_note": (
                  "Candidates run on this node with one worker per GPU. GPUs may be shared with other users, so FPS reflects concurrent load. Historical baseline FPS is reference only; no cross-node FPS delta is computed."
                  if plan.get("allow_shared_gpus") else
                  "Candidates run on this node with one worker per GPU. Historical baseline FPS is reference only; no cross-node FPS delta is computed."
              )}
    write_json(root / "summary.json", result)
    if rows:
        temporary = root / ".summary.csv.tmp"
        with temporary.open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(root / "summary.csv")
    return result


def write_comparison(plan):
    """Compare completed candidates and clearly labelled historical references."""
    summary = write_summary(plan, final=True)
    root = Path(plan["output"])
    baseline = summary["baseline_reference"]
    rows = [{"version": "original_v3", "step": 0, "source": "historical_baseline",
             "episodes": baseline["episodes"], "frames": baseline["frames"],
             **{key: baseline[key] for key in METRICS}}]
    rows.extend({"version": f"update_{row['step']}", "source": "current_run", **row}
                for row in summary["rows"])
    for historical in plan.get("historical_metrics", []):
        data = read_json(Path(historical["path"]))
        for key in ("model", "view", "manifest_sha256", "episodes", "frames", "protocol",
                    "metric_backbones", "metric_resolution", "native_resolution"):
            if data.get(key) != baseline.get(key):
                raise ValueError(f"Historical comparison protocols differ: {key}")
        if any(not math.isfinite(data[key]) for key in METRICS) or data["FPS"] <= 0:
            raise ValueError("Historical metrics contain invalid values")
        rows.append({"version": f"update_{historical['step']}_historical",
                     "step": historical["step"], "source": "historical_reference",
                     "episodes": data["episodes"], "frames": data["frames"],
                     **{key: data[key] for key in METRICS}})
    changes = {row["version"]: {
        key: {"delta": row[key] - baseline[key],
              "percent": 100 * (row[key] / baseline[key] - 1) if baseline[key] else None}
        for key in METRICS if key != "FPS"
    } for row in rows[1:]}
    result = {"rows": rows, "changes_vs_baseline": changes,
              "protocol_note": plan["protocol_note"], "fps_note": summary["fps_note"],
              "omitted_checkpoints": plan.get("omitted_checkpoints", [])}
    write_json(root / "comparison.json", result)
    temporary = root / ".comparison.csv.tmp"
    with temporary.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(root / "comparison.csv")
    return result


def inflight_complete(plan):
    """Adopt the existing evaluator without interrupting its metric subprocess."""
    root = Path(plan["output"])
    child = plan["inflight_evaluation"]
    if process_alive(child):
        return False
    destination = root / f"step_{child['step']:06d}"
    log = root / "logs" / f"step_{child['step']:06d}.log"
    marker = f"Complete: {destination / 'metrics.csv'}"
    if not log.is_file() or marker not in log.read_text(errors="replace"):
        raise RuntimeError("The adopted Cosmos evaluation exited before successful metric completion")
    if not (destination / "cosmos_metrics.json").is_file():
        raise RuntimeError("The adopted Cosmos evaluation has no final metric results")
    return True


def finish(plan):
    """Finish a user-shortened queue using the evaluator already in progress."""
    root = Path(plan["output"])
    if (plan.get("finish_after_step") != 1000
            or [row["step"] for row in plan["checkpoints"]] != [200, 600, 1000]
            or plan.get("inflight_evaluation", {}).get("step") != 1000):
        raise ValueError("This finish entry requires the reviewed 200/600/1000 queue")
    with (Path(plan["upstream"]) / "cosmos_checkpoint_queue.lock").open("a") as upstream_lock, (root / "queue.lock").open("a") as lock:
        fcntl.flock(upstream_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = read_json(root / "identity.json")
        if previous != plan["replaces_identity"] or process_alive(previous):
            raise RuntimeError("Previous supervisor has not released this queue")
        write_json(root / "identity.json", identity())

        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"Signal {signum}")

        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, interrupted)
        try:
            verify(plan)
            record(root, "waiting_for_inflight", step=1000, omitted_checkpoints=[2000])
            while not inflight_complete(plan):
                cancelled(root)
                time.sleep(2)
            verify(plan)
            write_comparison(plan)
            record(root, "complete", summary=str(root / "summary.csv"),
                   comparison=str(root / "comparison.csv"), omitted_checkpoints=[2000])
            (root / "exit_code").write_text("0\n")
        except (Exception, KeyboardInterrupt) as exc:
            status = "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed"
            record(root, status, error=f"{type(exc).__name__}: {exc}")
            (root / "exit_code").write_text("1\n")
            raise


def cancelled(root):
    if (root / "cancel.json").exists():
        raise KeyboardInterrupt("Queue cancelled by its cancel.json marker")


def gpu_total_memory():
    result = subprocess.run([
        "nvidia-smi", "--query-gpu=index,memory.total", "--format=csv,noheader,nounits",
    ], text=True, capture_output=True, check=True, timeout=20)
    return {row.split(",")[0].strip(): int(row.split(",")[1].strip())
            for row in result.stdout.splitlines() if row.strip()}


def wait_idle(plan, step):
    root, reported = Path(plan["output"]), 0
    while True:
        cancelled(root)
        usage, active = gpu_usage(), gpu_compute_pids(plan["gpus"])
        if plan.get("allow_shared_gpus"):
            ready = all(device in usage and device in plan["gpu_total_mib"]
                        and plan["gpu_total_mib"][device] - usage[device] >= plan["minimum_free_mib"]
                        for device in plan["gpus"])
        else:
            ready = not active and all(usage.get(device, 81920) <= 1024 for device in plan["gpus"])
        if ready:
            return
        if time.monotonic() - reported >= 30:
            record(root, "waiting_for_gpus", step=step, used_mib=usage, compute_pids=active)
            reported = time.monotonic()
        time.sleep(2)


def run(plan):
    from spatial_wovr_launch import stop_process
    from wm_eval import configure_environment

    root = Path(plan["output"])
    with (Path(plan["upstream"]) / "cosmos_checkpoint_queue.lock").open("a") as upstream_lock, (root / "queue.lock").open("a") as lock:
        fcntl.flock(upstream_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "identity.json").exists():
            raise RuntimeError("This queue already started; use a fresh output for another attempt")
        write_json(root / "identity.json", identity())

        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"Signal {signum}")

        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, interrupted)
        try:
            verify(plan)
            record(root, "waiting_for_wan_release" if plan.get("start_condition") == "stopped_predecessor" else "waiting_for_wan",
                   checkpoints=[row["step"] for row in plan["checkpoints"]])
            while not upstream_ready(plan):
                cancelled(root)
                time.sleep(2)
            os.environ.update(plan["environment"])
            configure_environment()
            env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", PYTHONUNBUFFERED="1",
                       CUDA_VISIBLE_DEVICES=",".join(plan["gpus"]))
            for key in ("RAY_ADDRESS", "RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
                env.pop(key, None)
            for checkpoint in plan["checkpoints"]:
                cancelled(root)
                verify(plan)
                wait_idle(plan, checkpoint["step"])
                record(root, "packaging", step=checkpoint["step"])
                packaged = package_checkpoint(root, checkpoint)
                destination = root / f"step_{checkpoint['step']:06d}"
                destination.mkdir()
                shutil.copyfile(root / "inputs/eval_manifest.json", destination / "eval_manifest.json")
                verify(plan)
                wait_idle(plan, checkpoint["step"])
                log = root / "logs" / f"step_{checkpoint['step']:06d}.log"
                record(root, "running", step=checkpoint["step"], log=str(log))
                with log.open("a") as stream:
                    child = subprocess.Popen(evaluation_command(plan, destination), cwd=PROJECT,
                        env=dict(env, COSMOS_WRIST_CHECKPOINT=str(packaged)),
                        stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                    process = read_process(child.pid)
                    write_json(root / "active_child.json", {"pid": child.pid,
                               "start_ticks": process.started, "step": checkpoint["step"]})
                    try:
                        while child.poll() is None:
                            cancelled(root)
                            time.sleep(2)
                        if child.returncode:
                            raise RuntimeError(f"Cosmos step {checkpoint['step']} evaluation exited {child.returncode}: {log}")
                    finally:
                        stop_process(child)
                summary = write_summary(plan)
                record(root, "checkpoint_complete", step=checkpoint["step"], completed=summary["completed_checkpoints"])
            write_summary(plan, final=True)
            record(root, "complete", summary=str(root / "summary.csv"))
            (root / "exit_code").write_text("0\n")
        except (Exception, KeyboardInterrupt) as exc:
            status = "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed"
            record(root, status, error=f"{type(exc).__name__}: {exc}")
            (root / "exit_code").write_text("1\n")
            raise


def notify(plan):
    root = Path(plan["output"])
    if Path("/proc/sys/kernel/random/boot_id").read_text().strip() != plan["boot_id"]:
        raise RuntimeError("Run this notifier on its recorded node")
    email_root = root / "outcome_email"
    email_root.mkdir(exist_ok=True)
    with (email_root / "notifier.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(root / "notifier_identity.json", identity())
        mailer = OutcomeMailer(plan, DEFAULT_CONFIG)
        while True:
            state = read_json(root / "state.json")
            worker = read_json(root / "identity.json")
            if worker and not process_alive(worker) and state.get("status") not in TERMINAL:
                state = read_json(root / "state.json")
                if state.get("status") not in TERMINAL:
                    state = {"status": "failed", "error": "Cosmos evaluation watchdog exited unexpectedly"}
            if state.get("status") in TERMINAL and "primary-outcome" not in mailer.events:
                title = "Cosmos 多权重评估完成" if state["status"] == "complete" else "Cosmos 多权重评估停止"
                body = (f"{title}\nUTC：{utc()}\n检查点：{[row['step'] for row in plan['checkpoints']]}\n"
                        f"验证集：{plan['evaluation_summary']}\n输出：{root}\n"
                        f"状态：{json.dumps(state, ensure_ascii=False)}\n")
                report = root / "comparison.csv" if (root / "comparison.csv").is_file() else root / "summary.csv"
                if report.is_file():
                    body += report.read_text()
                mailer.enqueue("primary-outcome", {"subject": f"[Spatial Cosmos] {title}",
                               "body": body, "outcome": state["status"]})
            mailer.send_due()
            sent = bool(mailer.events) and all(value.get("status") == "sent" for value in mailer.deliveries.values())
            mailer.persist(email_root / "state.json", {**mailer.status(), "status": "notified" if sent else "watching"})
            if sent:
                return
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("arm", "run", "notify", "finish"))
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--training-plan", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int, nargs="+", default=[200, 600, 1000, 2000])
    parser.add_argument("--start-immediately", action="store_true",
                        help="Arm after an explicit local Wan stop; skip Wan training/metrics completion")
    parser.add_argument("--allow-shared-gpus", action="store_true",
                        help="Allow concurrent GPU users when every selected GPU has at least 32 GiB free")
    args = parser.parse_args()
    if args.mode == "arm":
        if args.training_plan is None or args.output is None:
            parser.error("arm requires --training-plan and --output")
        arm(args)
    else:
        if args.plan is None:
            parser.error("run/notify/finish requires --plan")
        plan = read_json(args.plan)
        {"run": run, "notify": notify, "finish": finish}[args.mode](plan)


if __name__ == "__main__":
    main()
