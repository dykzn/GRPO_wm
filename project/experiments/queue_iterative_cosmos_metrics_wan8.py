"""Score Cosmos on its four GPUs while Wan trains, then reshard Wan and run GRPO.

This companion leaves the live, frozen 4+4 training code and plan unchanged.
Cosmos releases only its own four GPUs after 2k. Wan continues on four GPUs
through both evaluations. A native checkpoint-boundary restart gives Wan an
operational save request, so the eight-rank handoff uses its latest state after
metrics finish instead of rolling back the progress made during evaluation.
"""

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
import uuid

from queue_cosmos_first_wan8 import (
    METRICS, identity, latest_metrics, latest_wan_checkpoint, stop_owned_tree,
)
from queue_spatial_iterative_update import assert_idle, grpo_stage, verify as verify_source
from queue_spatial_wm_update import process_alive, sha256, utc, write_json
from spatial_followup_email import OutcomeMailer, read_json
from stop_spatial_grpo import read_process, signal_process, snapshot
from watch_training_email import DEFAULT_CONFIG

PROJECT = Path(__file__).resolve().parents[1]
CODE = [
    "queue_iterative_cosmos_metrics_wan8.py", "queue_cosmos_first_wan8.py",
    "wm_update_wan_reshard.py", "wm_update_wan_controlled.py", "wm_eval_reserved.py", "wm_eval.py",
    "wm_eval_predict.py", "wm_eval_metrics.py", "wm_eval_data.py",
    "spatial_wovr_launch.py", "wm_update_memory.py",
    "world_model_grpo/cosmos_wrist_adapter.py",
]
SAVED_STEPS = (1000, 2000)


def source_plan(plan):
    return read_json(Path(plan["source_plan"]))


def verify(plan):
    if sha256(Path(plan["source_plan"])) != plan["source_plan_sha256"]:
        raise RuntimeError("The live source plan changed")
    verify_source(source_plan(plan))
    for name, expected in plan["code"].items():
        if sha256(PROJECT / "experiments" / name) != expected:
            raise RuntimeError(f"Reviewed handoff code changed: {name}")
    for filename, expected in plan["inputs"].items():
        if sha256(Path(filename)) != expected:
            raise RuntimeError(f"Frozen handoff input changed: {filename}")


def emit(plan, status, **details):
    root = Path(plan["output"])
    value = dict(status=status, updated_utc=utc(), pid=os.getpid(), **details)
    write_json(root / "state.json", value)
    with (root / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(value) + "\n")
    if (root / "ownership_transferred.json").exists():
        original = Path(plan["source_output"])
        write_json(original / "state.json", dict(value, continuation_plan=str(root / "plan.json")))
        pointer = PROJECT / "runtime_artifacts/training_queue/iterative_policy40_current.json"
        current = read_json(pointer)
        if current.get("output") == str(original):
            current.pop("error", None)
            current.update(value, continuation_plan=str(root / "plan.json"),
                           screen=plan["screen"], notifier_screen=plan["notifier_screen"])
            write_json(pointer, current)
    print(json.dumps(value), flush=True)


def validate_holdout(evaluation, training):
    by_id = {row["trajectory_id"]: row for row in training["episodes"]}
    train_groups = {row["split_group"] for row in training["episodes"] if row["split"] == "train"}
    if len(evaluation["episodes"]) != 100:
        raise ValueError("Use the same 100 held-out episodes as the previous evaluation")
    for row in evaluation["episodes"]:
        current = by_id.get(row["trajectory_id"])
        if (current is None or current["split"] != "val" or row["split_group"] in train_groups
                or row["sha256"] != current["sha256"]):
            raise ValueError("A comparison episode is missing or overlaps current training")


def prepare(args):
    source = read_json(args.source_plan)
    verify_source(source)
    if source["wm_steps"] != 2000 or source["wm_save_interval"] != 1000:
        raise ValueError("This policy saves only the 1000 and 2000 step checkpoints")
    original = Path(source["output"])
    supervisor = read_json(original / "worker_identity.json")
    training = read_json(original / "active_child.json")
    current = read_json(original / "state.json")
    if current.get("status") != "running" or current.get("stage") != "train":
        raise RuntimeError("The original 4+4 update must still be training")
    for record in (supervisor, training):
        process = read_process(record["pid"])
        if (process is None or process.started != record["start_ticks"]
                or process.uid != os.geteuid()):
            raise RuntimeError("The recorded original process is no longer alive")
    if read_process(training["pid"]).ppid != supervisor["pid"]:
        raise ValueError("The original training is not a child of its recorded supervisor")
    groups = {}
    for model in ("wan", "cosmos"):
        matches = [p for p in snapshot().values() if p.ppid == training["pid"]
                   and "torch.distributed.run" in p.argv and "--model" in p.argv
                   and p.argv[p.argv.index("--model")+1] == model]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one original four-rank {model} torchrun group")
        groups[model] = identity(matches[0])
    notifications = [p for p in snapshot().values()
                     if str(PROJECT / "experiments/queue_spatial_iterative_update.py") in p.argv
                     and str(args.source_plan.resolve()) in p.argv and "notify" in p.argv]
    if len(notifications) != 1:
        raise RuntimeError("Expected one original CPU email notifier")
    proof = read_json(args.reshard_proof)
    if (proof.get("status") != "passed" or proof.get("source_world") != 4
            or proof.get("target_world") != 8 or proof.get("effective_batch") != 16
            or not proof.get("model_and_optimizer_restored_exactly")
            or not proof.get("eight_rank_checkpoint_saved") or proof.get("cuda_initialized")):
        raise ValueError("A successful CPU model/Adam four-to-eight migration proof is required")
    control_proof = read_json(args.control_proof)
    if (control_proof.get("status") != "passed" or control_proof.get("world_size") != 4
            or not control_proof.get("native_model_adam_rng_checkpoint")
            or not control_proof.get("all_ranks_stopped_at_same_boundary")
            or control_proof.get("cuda_initialized")):
        raise ValueError("A successful four-rank CPU controlled-checkpoint proof is required")
    wm = Path(source["wm_output"])
    evaluation = read_json(args.eval_manifest)
    manifest = read_json(wm / "manifest.json")
    validate_holdout(evaluation, manifest)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "inputs").mkdir()
    (root / "logs").mkdir()
    for source_file, name in [(args.eval_manifest, "eval_manifest.json"),
                              (args.reshard_proof, "reshard_cpu_proof.json"),
                              (args.control_proof, "control_cpu_proof.json")]:
        shutil.copyfile(source_file, root / "inputs" / name)
    for directory in ("video_metrics", "train4", "train8"):
        target = wm / root.name / directory
        target.mkdir(parents=True, exist_ok=False)
        (root / directory).symlink_to(target.resolve(), target_is_directory=True)
    plan = dict(
        schema="spatial-iterative-cosmos-metrics4-concurrent-wan8-grpo60-v2", created_utc=utc(),
        output=str(root), source_output=str(original), source_plan=str(args.source_plan.resolve()),
        source_plan_sha256=sha256(args.source_plan), original_supervisor=supervisor,
        original_training=training, original_notifier=identity(notifications[0]), original_groups=groups,
        wm_output=str(wm), python=source["python"], seed=source["seed"],
        steps=2000, save_interval=1000, checkpoint_steps=list(SAVED_STEPS),
        evaluation_steps=list(SAVED_STEPS), evaluation_gpus=source["model_allocation"]["cosmos"],
        wan_gpus=source["model_allocation"]["wan"], wan_control_token=uuid.uuid4().hex,
        gpus=source["gpus"], old_ranks=4, new_ranks=8,
        old_accumulation=4, new_accumulation=2, effective_batch=16,
        manifest_sha256=manifest["manifest_sha256"],
        evaluation_manifest_sha256=evaluation["manifest_sha256"],
        evaluation_summary=evaluation["summary"],
        reference_selection="Unchanged previous 100 validation episodes, verified held out in current 1000-episode manifest",
        screen="spatial-policy40-metrics4-wan8", notifier_screen="spatial-policy40-metrics4-mail",
        code={name: sha256(PROJECT / "experiments" / name) for name in CODE},
    )
    plan["inputs"] = {str(path): sha256(path) for path in
                      [root / "inputs/eval_manifest.json", root / "inputs/reshard_cpu_proof.json",
                       root / "inputs/control_cpu_proof.json", wm / "manifest.json"]}
    verify(plan)
    write_json(root / "plan.json", plan)
    emit(plan, "prepared", stage="wait_checkpoints")
    pointer = PROJECT / "runtime_artifacts/training_queue/iterative_policy40_current.json"
    current = read_json(pointer)
    if current.get("output") == str(original):
        current["pending_policy_update"] = dict(plan=str(root / "plan.json"),
            order=["finish_cosmos2000", "cosmos1000_metrics4_parallel_wan4", "cosmos2000_metrics4_parallel_wan4", "save_latest_wan", "wan_resume8_to2000", "grpo40_to60"],
            checkpoint_steps=list(SAVED_STEPS), updated_utc=utc())
        write_json(pointer, current)
    print(str(root / "plan.json"), flush=True)


def restart(args):
    """Retry a failed pre-handoff evaluation without replaying data/WM training."""
    previous = read_json(args.plan)
    old = Path(previous["output"])
    if read_json(old / "state.json").get("status") != "failed":
        raise ValueError("Restart requires a failed previous attempt")
    if (old / "ownership_transferred.json").exists():
        raise ValueError("This restart only supports failures before the Wan controller handoff")
    for name in ("worker_identity.json", "notifier_identity.json", "active_child.json", "wan_child.json"):
        worker = read_json(old / name)
        if worker and process_alive(worker):
            raise RuntimeError(f"The failed attempt still has a live process: {name}")
    if sha256(Path(previous["source_plan"])) != previous["source_plan_sha256"]:
        raise ValueError("The original training plan changed")
    verify_source(source_plan(previous))
    for name, expected in previous["code"].items():
        if name != Path(__file__).name and sha256(PROJECT / "experiments" / name) != expected:
            raise ValueError(f"Unexpected training/evaluation code change: {name}")
    for name, expected in previous["inputs"].items():
        if sha256(Path(name)) != expected:
            raise ValueError(f"Frozen input changed: {name}")
    for record in (previous["original_supervisor"], previous["original_training"], previous["original_groups"]["wan"]):
        if not process_alive(record):
            raise RuntimeError("Original four-rank Wan must still be running for this retry")
    if process_alive(previous["original_groups"]["cosmos"]):
        raise RuntimeError("The completed Cosmos group has not released its GPUs")
    root = args.output.resolve()
    (root / "inputs").mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    for name in ("eval_manifest.json", "reshard_cpu_proof.json", "control_cpu_proof.json"):
        shutil.copyfile(old / "inputs" / name, root / "inputs" / name)
    for name in ("video_metrics", "train4", "train8"):
        destination = Path(previous["wm_output"]) / root.name / name
        destination.mkdir(parents=True, exist_ok=False)
        (root / name).symlink_to(destination.resolve(), target_is_directory=True)
    plan = dict(previous, output=str(root), created_utc=utc(), restart_of=str(old),
        previous_plan=str(args.plan.resolve()), previous_plan_sha256=sha256(args.plan),
        restart_reason="Give the evaluation supervisor its four CUDA-visible GPUs; preserve CPU-only queue supervisor",
        cosmos_already_released=True, wan_control_token=uuid.uuid4().hex,
        screen="spatial-policy40-metrics4-retry1", notifier_screen="spatial-policy40-metrics4-mail-r1",
        code={name: sha256(PROJECT / "experiments" / name) for name in CODE})
    manifest = Path(plan["wm_output"]) / "manifest.json"
    plan["inputs"] = {str(path): sha256(path) for path in [manifest, args.plan.resolve(),
        root / "inputs/eval_manifest.json", root / "inputs/reshard_cpu_proof.json", root / "inputs/control_cpu_proof.json"]}
    verify(plan)
    if not cosmos_ready(plan):
        raise RuntimeError("Both completed Cosmos 1k/2k checkpoints are required")
    checkpoint = ready_checkpoint(plan)
    if checkpoint is None:
        raise RuntimeError("A full original Wan checkpoint is required before retry")
    plan["wan_restart_checkpoint"] = str(checkpoint)
    plan["wan_latest_logged_step_before_restart"] = latest_metrics(Path(plan["wm_output"]) / "train/wan/metrics.jsonl").get("step")
    write_json(root / "plan.json", plan)
    emit(plan, "prepared", stage="retry_cosmos_metrics4", checkpoint=str(checkpoint))
    pointer = PROJECT / "runtime_artifacts/training_queue/iterative_policy40_current.json"
    current = read_json(pointer)
    current["pending_policy_update"] = dict(plan=str(root / "plan.json"), restart_of=str(old),
        order=["cosmos1000_metrics4_parallel_wan4", "cosmos2000_metrics4_parallel_wan4", "save_latest_wan", "wan_resume8_to2000", "grpo40_to60"],
        checkpoint_steps=list(SAVED_STEPS), updated_utc=utc())
    write_json(pointer, current)
    print(str(root / "plan.json"), flush=True)


def cosmos_checkpoint(plan, step):
    return (Path(plan["wm_output"]) / "train/cosmos/spatial_wmevo/cosmos_wrist_v3/train/checkpoints"
            / f"iter_{step:09d}")


def cosmos_ready(plan):
    for step in SAVED_STEPS:
        path = cosmos_checkpoint(plan, step)
        record = read_json(path / "wm_update_complete.json")
        if not record:
            return False
        if (record.get("step") != step or record.get("world_size") != 4
                or record.get("identity", {}).get("manifest") != plan["manifest_sha256"]):
            raise ValueError("Unexpected Cosmos checkpoint step, layout or training data")
        if not all((path / name / ".metadata").is_file() for name in ("model", "optim", "scheduler", "trainer")):
            raise ValueError("Cosmos checkpoint is missing native model/optimizer state")
        if not (path / "master_weights.pt").is_file():
            raise ValueError("Cosmos checkpoint is missing FP32 master weights")
        if not all((path / f"rng_rank{rank}.pt").is_file() for rank in range(4)):
            raise ValueError("Cosmos checkpoint is missing a rank RNG state")
    if plan.get("cosmos_already_released"):
        if process_alive(plan["original_groups"]["cosmos"]):
            raise RuntimeError("Retry expected the completed Cosmos GPUs to be released")
        return True
    handoff = Path(plan["source_output"]) / "training_handoff"
    for rank in range(4):
        receipt = read_json(handoff / f"cosmos_rank{rank}.json")
        if not receipt:
            return False
        worker = read_process(receipt["pid"])
        if (receipt.get("model") != "cosmos" or receipt.get("rank") != rank
                or receipt.get("phase") != "training_complete_cache_retained"
                or receipt.get("reserved_gib", 0) < 1 or worker is None
                or worker.uid != os.geteuid() or "cosmos" not in worker.argv):
            raise RuntimeError("Cosmos did not finish with its computing-worker cache retained")
    return True


def ready_checkpoint(plan):
    """A committed original Wan checkpoint; Cosmos may already be evaluating."""
    try:
        checkpoint = latest_wan_checkpoint(Path(plan["wm_output"]) / "train/wan", plan["manifest_sha256"])
    except ValueError as error:
        if str(error).startswith("No complete, exported and logged"):
            return None
        raise
    record = read_json(checkpoint / "complete.json")
    if record.get("step") not in SAVED_STEPS or record.get("world_size") != 4:
        raise ValueError("Wan handoff requires an original four-rank 1000/2000 full checkpoint")
    return checkpoint


def release_cosmos(plan):
    """Pause only the CPU controller and release the finished Cosmos workers."""
    root = Path(plan["output"])
    process = read_process(plan["original_supervisor"]["pid"])
    if process is None or process.started != plan["original_supervisor"]["start_ticks"]:
        raise RuntimeError("Original supervisor exited before Cosmos evaluation")
    signal_process(process, signal.SIGSTOP)
    write_json(root / "controller_paused.json", dict(utc=utc(), identity=identity(process)))
    if plan.get("cosmos_already_released"):
        emit(plan, "stage_complete", stage="cosmos_released_for_metrics4", wan_continues=True,
             note="Reusing already completed 1000/2000 checkpoints after preflight-only failure")
        return
    release = Path(plan["source_output"]) / "training_handoff/release.json"
    write_json(release, dict(reason="Release finished Cosmos only; Wan continues on four GPUs", utc=utc()))
    deadline = time.monotonic() + 90
    while process_alive(plan["original_groups"]["cosmos"]):
        if not process_alive(plan["original_groups"]["wan"]):
            raise RuntimeError("Wan stopped during the Cosmos-only CUDA-cache handoff")
        if time.monotonic() >= deadline:
            raise RuntimeError("Finished Cosmos did not release its four GPUs")
        time.sleep(.25)
    # Keep the original live owner and restore its hold for Wan if it reaches
    # 2k during metrics. The old CPU controller remains paused and cannot start
    # packaging or GRPO while the evaluation uses the other four GPUs.
    release.replace(root / "cosmos_cache_release.json")
    emit(plan, "stage_complete", stage="cosmos_released_for_metrics4", wan_continues=True)


def recover_original(plan):
    """A companion failure before completed ownership transfer must not strand the CPU supervisor."""
    root = Path(plan["output"])
    if not (root / "controller_paused.json").exists() or (root / "ownership_transferred.json").exists():
        return
    process = read_process(plan["original_supervisor"]["pid"])
    if process and process.started == plan["original_supervisor"]["start_ticks"]:
        signal_process(process, signal.SIGCONT)
        write_json(root / "original_controller_resumed.json", dict(utc=utc(), identity=identity(process)))


def take_ownership(plan, checkpoint):
    verify(plan)
    root, original = Path(plan["output"]), Path(plan["source_output"])
    process = read_process(plan["original_supervisor"]["pid"])
    if process is None or process.started != plan["original_supervisor"]["start_ticks"]:
        raise RuntimeError("Original supervisor exited before the controlled handoff")
    write_json(root / "selected_wan_checkpoint.json", dict(path=str(checkpoint),
        record=read_json(checkpoint / "complete.json"), utc=utc()))
    signal_process(process, signal.SIGSTOP)
    write_json(root / "controller_paused.json", dict(utc=utc(), identity=identity(process)))
    emit(plan, "running", stage="replace_original_controller", checkpoint=str(checkpoint))
    # The old supervisor owns cleanup of its training group. Pause its CPU loop
    # before deliberately stopping that group, otherwise it treats the switch
    # as a training failure and may clean up the replacement GPUs.
    stop_owned_tree(plan["original_training"])
    stop_owned_tree(plan["original_notifier"], immediate=True)
    signal_process(process, signal.SIGKILL)
    deadline = time.monotonic() + 10
    while process_alive(plan["original_supervisor"]):
        if time.monotonic() >= deadline:
            raise RuntimeError("Original CPU supervisor did not release its workflow lock")
        time.sleep(.1)
    lock = (original / "workflow.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    own = identity(read_process(os.getpid()))
    for filename in ("worker_identity.json", "primary_identity.json"):
        shutil.copyfile(original / filename, root / ("superseded_" + filename))
        write_json(original / filename, own)
    write_json(root / "ownership_transferred.json", dict(utc=utc(), identity=own,
        reason="User requested Cosmos 1k/2k metrics before eight-rank Wan and GRPO continuation"))
    emit(plan, "stage_complete", stage="checkpoint_handoff", checkpoint=str(checkpoint))
    return lock


def package_cosmos(plan, step):
    source = cosmos_checkpoint(plan, step)
    root = Path(plan["output"]) / "cosmos_packages" / f"step_{step}"
    target = root / source.name
    (target / "model").mkdir(parents=True, exist_ok=False)
    rows = []
    for path in sorted((source / "model").rglob("*")):
        if path.is_file():
            relative = path.relative_to(source / "model")
            linked = target / "model" / relative
            linked.parent.mkdir(parents=True, exist_ok=True)
            linked.symlink_to(path.resolve())
            rows.append(dict(path=str(Path(source.name) / "model" / relative),
                             size=path.stat().st_size, sha256=sha256(path)))
    if not rows or not (target / "model/.metadata").is_file():
        raise ValueError("Incomplete Cosmos inference package")
    tree = hashlib.sha256("".join(f"{row['sha256']}  {row['path']}\n" for row in rows).encode()).hexdigest()
    write_json(root / "checkpoint_manifest.json", dict(checkpoint=source.name,
        run_name="spatial_wmevo", format="pytorch-distributed-checkpoint", files=rows,
        file_count=len(rows), total_bytes=sum(row["size"] for row in rows), tree_sha256=tree))
    return target


def evaluate_command(plan, step):
    root = Path(plan["output"])
    return [plan["python"], str(PROJECT / "experiments/wm_eval_reserved.py"), "run",
        "--model", "cosmos", "--manifest", str(root / "inputs/eval_manifest.json"),
        "--output", str(root / "video_metrics" / f"cosmos_step{step}"),
        "--gpus", ",".join(plan["evaluation_gpus"]), "--steps", "5",
        "--warmup", "3", "--seed", str(plan["seed"]), "--metric-batch", "8"]


def evaluation_environment(plan, env, checkpoint):
    """The CPU queue is hidden; evaluation preflight must see its four GPUs."""
    return dict(env, COSMOS_WRIST_CHECKPOINT=str(checkpoint),
                CUDA_VISIBLE_DEVICES=",".join(plan["evaluation_gpus"]))


def wan_command(plan, checkpoint):
    source, root = source_plan(plan), Path(plan["output"])
    return [plan["python"], "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
        "--nproc_per_node=8", str(PROJECT / "experiments/wm_update_wan_reshard.py"),
        "--manifest", str(Path(plan["wm_output"]) / "manifest.json"),
        "--settings", str(Path(source["output"]) / "inputs/settings.json"),
        "--resume", str(checkpoint), "--output", str(root / "train8/wan"),
        "--steps", "2000", "--save-interval", "1000", "--accumulation", "2",
        "--lr", "1e-6", "--seed", str(plan["seed"]),
        "--reserve-memory", "--memory-headroom-gib", "3.5"]


def wan_four_command(plan, checkpoint):
    root, source = Path(plan["output"]), source_plan(plan)
    return [plan["python"], "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
        "--nproc_per_node=4", str(PROJECT / "experiments/wm_update_wan_controlled.py"),
        "--manifest", str(Path(plan["wm_output"]) / "manifest.json"),
        "--settings", str(Path(source["output"]) / "inputs/settings.json"),
        "--resume", str(checkpoint), "--output", str(root / "train4/wan"),
        "--control-dir", str(root / "wan_control"), "--control-token", plan["wan_control_token"],
        "--owner-identity", json.dumps(identity(read_process(os.getpid()))),
        "--steps", "2000", "--save-interval", "1000", "--accumulation", "4",
        "--lr", "1e-6", "--seed", str(plan["seed"]),
        "--reserve-memory", "--memory-headroom-gib", "3.5"]


def maintain_wan(plan, env, running):
    """Give Wan a save-request hook at its next committed native checkpoint."""
    from wm_update import runtime

    child = running.get("wan_child")
    if child is not None:
        if child.poll() is not None:
            raise RuntimeError(f"Parallel four-rank Wan exited unexpectedly: {child.returncode}")
        return
    if not process_alive(plan["original_training"]):
        raise RuntimeError("Original Wan supervisor exited during Cosmos metrics")
    checkpoint = ready_checkpoint(plan)
    if checkpoint is None or read_json(checkpoint / "complete.json")["step"] == 2000:
        return
    root = Path(plan["output"])
    phase = read_json(root / "state.json").get("stage", "cosmos_metrics4_parallel_wan4")
    running["ownership_lock"] = take_ownership(plan, checkpoint)
    runtime(root / "train4")
    env.update({key: value for key, value in os.environ.items()
                if key in ("PYTHONPATH", "LIBERO_CONFIG_PATH", "WM_EVAL_RUNTIME_READY")})
    assert_idle(plan["wan_gpus"])
    stream = (root / "logs/wan_parallel4.log").open("a")
    running["wan_stream"] = stream
    child = subprocess.Popen(wan_four_command(plan, checkpoint), cwd=PROJECT,
        env=dict(env, CUDA_VISIBLE_DEVICES=",".join(plan["wan_gpus"]), NCCL_SOCKET_IFNAME="lo"),
        stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    running["wan_child"] = child
    write_json(root / "wan_child.json", dict(identity(read_process(child.pid)),
        resumed_step=1000, gpus=plan["wan_gpus"], native_output=str(root / "train4/wan")))
    emit(plan, "stage_complete", stage="wan_save_request_enabled4", resumed_step=1000,
         note="Wan keeps training during four-GPU Cosmos metrics; full optimizer/RNG restored")
    emit(plan, "running", stage=phase, wan_parallel_gpus=plan["wan_gpus"],
         wan_parallel_output=str(root / "train4/wan"))


def controlled_checkpoint(plan):
    root = Path(plan["output"])
    rows = [read_json(root / f"wan_control/finished_rank{rank}.json") for rank in range(4)]
    if not all(rows):
        return None
    step = rows[0].get("step")
    if not isinstance(step, int) or not 1000 < step <= 2000:
        raise ValueError("Invalid Wan handoff optimizer step")
    for rank, row in enumerate(rows):
        if (row.get("rank") != rank or row.get("step") != step
                or row.get("reserved_gib", 0) < 1 or not process_alive(row)):
            raise RuntimeError("A Wan worker did not retain its complete-checkpoint CUDA cache")
    checkpoint = latest_wan_checkpoint(root / "train4/wan", plan["manifest_sha256"])
    record = read_json(checkpoint / "complete.json")
    if record.get("step") != step or record.get("world_size") != 4:
        raise ValueError("Wan handoff has no matching full checkpoint and runtime export")
    if any(Path(row["checkpoint"]) != checkpoint for row in rows):
        raise ValueError("Wan ranks recorded different handoff checkpoints")
    return checkpoint


def finish_parallel_wan(plan, env, running):
    """Request a fresh full checkpoint after metrics, then release four ranks."""
    root = Path(plan["output"])
    child = running.get("wan_child")
    if child is None:
        # If both metrics finished before the first native 1k save, let that
        # uninterrupted original worker reach its first full checkpoint.
        emit(plan, "waiting_for_checkpoint", stage="wan_save_for_resize")
        while True:
            if not process_alive(plan["original_training"]):
                raise RuntimeError("Original Wan stopped before its handoff checkpoint")
            checkpoint = ready_checkpoint(plan)
            if checkpoint is not None:
                break
            time.sleep(.25)
        running["ownership_lock"] = take_ownership(plan, checkpoint)
    else:
        request = root / "wan_control/checkpoint_request.json"
        write_json(request, dict(token=plan["wan_control_token"], manifest_sha256=plan["manifest_sha256"],
            utc=utc(), reason="Both Cosmos metric runs completed; save latest full Wan state before eight-rank resume"))
        emit(plan, "running", stage="wan_save_for_resize", checkpoint_request=str(request))
        while True:
            if child.poll() is not None:
                raise RuntimeError(f"Wan exited before saving the requested checkpoint: {child.returncode}")
            checkpoint = controlled_checkpoint(plan)
            if checkpoint is not None:
                break
            time.sleep(.5)
        write_json(root / "wan_control/release.json", dict(utc=utc(), checkpoint=str(checkpoint)))
        if child.wait(timeout=90):
            raise RuntimeError("Four-rank Wan did not exit cleanly after its completed checkpoint")
    write_json(root / "selected_wan_checkpoint.json", dict(path=str(checkpoint),
        record=read_json(checkpoint / "complete.json"), utc=utc(), handoff_requested_after_metrics=True,
        save_policy="requested_at_next_update" if child is not None else "next_native_checkpoint"))
    emit(plan, "stage_complete", stage="wan_save_for_resize", checkpoint=str(checkpoint),
         step=read_json(checkpoint / "complete.json")["step"])
    return checkpoint


def stage(plan, name, command, env, monitor=None):
    from spatial_wovr_launch import stop_process

    verify(plan)
    root = Path(plan["output"])
    log = root / "logs" / f"{name}.log"
    emit(plan, "running", stage=name, log=str(log))
    with log.open("a") as stream:
        child = subprocess.Popen(command, cwd=PROJECT, env=env, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        try:
            write_json(root / "active_child.json", dict(identity(read_process(child.pid)), stage=name))
            while child.poll() is None:
                if monitor is not None:
                    monitor()
                time.sleep(.5)
            code = child.returncode
            if code:
                raise RuntimeError(f"{name} exited {code}: {log}")
        finally:
            stop_process(child)
    emit(plan, "stage_complete", stage=name, log=str(log))


def summarize(plan):
    rows = []
    reference = None
    for step in SAVED_STEPS:
        result = read_json(Path(plan["output"]) / f"video_metrics/cosmos_step{step}/cosmos_metrics.json")
        if (result.get("manifest_sha256") != plan["evaluation_manifest_sha256"]
                or result.get("episodes") != plan["evaluation_summary"]["episodes"]
                or result.get("frames") != plan["evaluation_summary"]["future_frames"]
                or result.get("model") != "cosmos" or result.get("metric_shards") != len(plan["evaluation_gpus"])
                or not all(math.isfinite(result[key]) for key in METRICS) or result["FPS"] <= 0):
            raise ValueError("Cosmos metrics do not cover the frozen four-shard evaluation")
        if reference is not None and any(result[key] != reference[key] for key in
                ("protocol", "metric_backbones", "metric_resolution", "horizon_min", "horizon_max")):
            raise ValueError("The two Cosmos metric protocols differ")
        reference = result
        rows.append(dict(step=step, **{key: result[key] for key in METRICS}))
    root = Path(plan["output"])
    write_json(root / "cosmos_comparison.json", dict(rows=rows,
        manifest_sha256=plan["evaluation_manifest_sha256"], summary=plan["evaluation_summary"]))
    with (root / "cosmos_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    emit(plan, "stage_complete", stage="cosmos_comparison", comparison=str(root / "cosmos_comparison.csv"))


def run(plan):
    from spatial_followup_watchdogs import cleanup_workload, finished_grpo, run_child
    from spatial_wovr_launch import stop_process
    from wm_update import environment, runtime

    root = Path(plan["output"])
    running = {}
    with (root / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "worker_identity.json").exists():
            raise FileExistsError("This continuation already ran; do not replay training or evaluations")
        write_json(root / "worker_identity.json", identity(read_process(os.getpid())))
        try:
            verify(plan)
            emit(plan, "waiting_for_checkpoints", stage="wait_cosmos2000", wan_continues=True)
            while True:
                if (not process_alive(plan["original_supervisor"])
                        or not process_alive(plan["original_training"])):
                    raise RuntimeError("Original world-model training stopped before the handoff")
                if cosmos_ready(plan):
                    break
                time.sleep(.5)
            source = source_plan(plan)
            os.environ.update(source["environment"], CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1")
            environment()
            env = dict(os.environ, SPATIAL_FOLLOWUP_CHAIN=source["output"], SPATIAL_FOLLOWUP_ROLE="primary_workload")
            for name in ("WM_UPDATE_HANDOFF_ROOT", "WM_UPDATE_HANDOFF_OWNER", "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
                env.pop(name, None)
            # Prepare the CPU-only serving manifests while Cosmos still owns
            # its cache. Release only its group immediately before evaluation.
            models = {step: package_cosmos(plan, step) for step in SAVED_STEPS}
            release_cosmos(plan)
            for step in SAVED_STEPS:
                maintain_wan(plan, env, running)
                assert_idle(plan["evaluation_gpus"])
                stage(plan, f"cosmos{step}_metrics4_parallel_wan4", evaluate_command(plan, step),
                      evaluation_environment(plan, env, models[step]),
                      monitor=lambda: maintain_wan(plan, env, running))
            summarize(plan)
            checkpoint = finish_parallel_wan(plan, env, running)
            if read_json(checkpoint / "complete.json")["step"] < plan["steps"]:
                assert_idle(plan["gpus"])
                runtime(root / "train8")
                env.update({key: value for key, value in os.environ.items()
                            if key in ("PYTHONPATH", "LIBERO_CONFIG_PATH", "WM_EVAL_RUNTIME_READY")})
                stage(plan, "wan_train8", wan_command(plan, checkpoint),
                      dict(env, CUDA_VISIBLE_DEVICES=",".join(plan["gpus"]), NCCL_SOCKET_IFNAME="lo"))
                checkpoint = latest_wan_checkpoint(root / "train8/wan", plan["manifest_sha256"])
            if read_json(checkpoint / "complete.json")["step"] != 2000:
                raise RuntimeError("Wan did not save a complete final 2000-step checkpoint")
            stage(plan, "package", [plan["python"], str(PROJECT / "experiments/wm_update.py"), "package",
                "--manifest", str(Path(plan["wm_output"]) / "manifest.json"),
                "--wan-checkpoint", str(checkpoint), "--cosmos-checkpoint", str(cosmos_checkpoint(plan, 2000)),
                "--output", str(Path(source["output"]) / "model_pair")], dict(env, CUDA_VISIBLE_DEVICES=""))
            source.update(screen=plan["screen"], notifier_screen=plan["notifier_screen"])
            command, grpo_env, output, target = grpo_stage(source)
            assert_idle(plan["gpus"])
            write_json(Path(source["output"]) / "grpo_launch.json", dict(command=command, output=str(output), target_step=target))
            pointer = PROJECT / "runtime_artifacts/training_queue/grpo_latestwm_current.json"
            latest = dict(status="launching", output=source["output"], plan=str(root / "plan.json"),
                grpo_output=str(output), resume_step=40, target_step=60, last_completed_step=40,
                wan_step=2000, cosmos_step=2000, cosmos_recipe="wrist-v3-reconstructed-v1",
                pair_manifest=str(Path(source["output"]) / "model_pair/pair.json"),
                gpus=[int(g) for g in plan["gpus"]], gpu_count=8,
                screen=plan["screen"], notifier_screen=plan["notifier_screen"], updated_utc=utc())
            write_json(pointer, latest)
            emit(plan, "running", stage="grpo", grpo_output=str(output), target_step=target)
            run_child(source, "primary", "grpo", command, grpo_env, resume=source["policy_checkpoint"],
                      target_step=target, grpo_output=str(output), updated_world_model=True)
            if not finished_grpo(output, target):
                raise RuntimeError("Continuation did not save the full final GRPO 60 checkpoint")
            cleanup_workload(source, "primary")
            latest.update(status="complete", last_completed_step=target, updated_utc=utc())
            write_json(pointer, latest)
            emit(plan, "complete", stage="grpo", grpo_output=str(output), target_step=target)
        except BaseException as error:
            recover_original(plan)
            if (root / "ownership_transferred.json").exists():
                cleanup_workload(source_plan(plan), "primary")
            previous = read_json(root / "state.json")
            emit(plan, "failed", stage=previous.get("stage"), error=f"{type(error).__name__}: {error}")
            raise
        finally:
            if running.get("wan_child") is not None:
                stop_process(running["wan_child"])
            if running.get("wan_stream") is not None:
                running["wan_stream"].close()
            if running.get("ownership_lock") is not None:
                running["ownership_lock"].close()


def notify(plan):
    root = Path(plan["output"])
    with (root / "notifier.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(root / "notifier_identity.json", identity(read_process(os.getpid())))
        mailer = OutcomeMailer(plan, DEFAULT_CONFIG)
        for file in (root / "outcome_email/outbox").glob("*.json"):
            mailer.events[file.stem] = read_json(file)
            mailer.deliveries[file.stem] = read_json(root / "outcome_email/delivery" / file.name)
        mailer.enqueue("armed", dict(outcome="armed", subject="[Spatial] Cosmos四卡指标与Wan四卡并行→保存→Wan八卡→GRPO队列已启用",
            body=f"Cosmos只保留1000/2000步。Cosmos到2000后，四卡评两份权重，同时Wan另四卡继续训练；指标完成后保存Wan最新完整状态，再八卡续训至2000，最后GRPO40→60。\n计划：{root / 'plan.json'}"))
        while True:
            result = read_json(root / "state.json")
            worker = read_json(root / "worker_identity.json")
            unexpected_exit = False
            if worker and not process_alive(worker) and result.get("status") not in {"complete", "failed"}:
                unexpected_exit = True
                result = dict(status="failed", stage=result.get("stage"), error="Continuation supervisor exited unexpectedly")
            events = root / "events.jsonl"
            for line in events.read_text().splitlines() if events.exists() else []:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get("status") == "stage_complete":
                    name = event["stage"]
                    mailer.enqueue("stage-" + name, dict(outcome="complete", subject=f"[Spatial 接续] {name} 完成",
                        body=f"阶段：{name}\n输出：{root}\n{json.dumps(event, ensure_ascii=False)}"))
            terminal = result.get("status") in {"complete", "failed"}
            if terminal:
                mailer.enqueue("primary-outcome", dict(outcome=result["status"],
                    subject=f"[Spatial 接续] Cosmos评估/Wan八卡/GRPO {result['status']}",
                    body=f"输出：{root}\n{json.dumps(result, ensure_ascii=False)}"))
            mailer.send_due()
            sent = terminal and mailer.deliveries.get("primary-outcome", {}).get("status") == "sent"
            mailer.persist(root / "outcome_email/state.json", dict(status="notified" if sent else "watching",
                updated_utc=utc(), deliveries=mailer.deliveries))
            # Notify before potentially slow orphan cleanup. Recover only
            # after the companion's metric workers released their GPUs.
            if unexpected_exit and not (root / "orphan_cleanup.json").exists():
                for filename in ("active_child.json", "wan_child.json"):
                    child = read_json(root / filename)
                    if child:
                        stop_owned_tree(child)
                recover_original(plan)
                write_json(root / "orphan_cleanup.json", dict(utc=utc(), status="complete"))
            if sent and all(row.get("status") == "sent" for row in mailer.deliveries.values()):
                return
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "restart", "verify", "run", "notify"])
    for name in ("plan", "source-plan", "eval-manifest", "reshard-proof", "control-proof", "output"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    if args.mode == "restart":
        if args.plan is None or args.output is None:
            parser.error("restart requires the failed --plan and a new --output")
        restart(args)
        return
    if args.mode == "prepare":
        if not all((args.source_plan, args.eval_manifest, args.reshard_proof, args.control_proof, args.output)):
            parser.error("prepare requires source plan, eval manifest, both CPU proofs and a new output")
        prepare(args)
        return
    if args.plan is None:
        parser.error("--plan is required")
    plan = read_json(args.plan)
    if args.mode == "run":
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(signum, lambda number, frame: (_ for _ in ()).throw(KeyboardInterrupt(f"Signal {number}")))
        run(plan)
    elif args.mode == "notify":
        notify(plan)
    else:
        verify(plan)
        print("Continuation verified; no GPU training/evaluation was started")


if __name__ == "__main__":
    main()
