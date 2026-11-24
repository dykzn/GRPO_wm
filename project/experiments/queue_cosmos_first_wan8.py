"""Score finished Cosmos, then migrate the latest complete Wan update to eight GPUs."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from queue_spatial_wm_update import (
    gpu_compute_pids, gpu_usage, process_alive, record, sha256, utc, verify_plan, write_json,
)
from spatial_followup_email import OutcomeMailer, read_json
from stop_spatial_grpo import read_process, signal_process, snapshot
from watch_training_email import DEFAULT_CONFIG

PROJECT = Path(__file__).resolve().parents[1]
CODE = [
    "queue_cosmos_first_wan8.py", "wm_update_wan_reshard.py", "wm_eval.py",
    "wm_eval_predict.py", "wm_eval_data.py", "wm_eval_metrics.py", "wm_update.py", "wm_update_wan.py",
    "wm_update_data.py", "wm_update_accept.py", "spatial_wovr_launch.py",
    "world_model_grpo/cosmos_wrist_adapter.py",
]
METRICS = ("FPS", "LPIPS", "FID", "FVD_variable_length", "FloLPIPS")


def identity(process):
    return {"pid": process.pid, "start_ticks": process.started}


def latest_metrics(path):
    if not path.is_file():
        return {}
    for line in reversed(path.read_text().splitlines()):
        try:
            return json.loads(line)
        except ValueError:
            continue
    return {}


def latest_wan_checkpoint(directory: Path, manifest: str):
    candidates = []
    logged = int(latest_metrics(directory / "metrics.jsonl").get("step", 0))
    for path in directory.glob("step_*"):
        complete = read_json(path / "complete.json")
        exported = read_json(path / "runtime/wm_update.json")
        step = complete.get("step", -1)
        if (complete.get("identity", {}).get("manifest") == manifest
                and exported.get("manifest_sha256") == manifest
                and exported.get("step") == step and 0 < step <= logged
                and (path / "state/.metadata").is_file()
                and (path / "runtime/model-00001.safetensors").is_file()
                and (path / "runtime/Wan2.2_VAE.pth").is_file()
                and all((path / f"rng_rank{rank}.pt").is_file()
                        for rank in range(complete.get("world_size", 0)))):
            candidates.append((step, path))
    if not candidates:
        raise ValueError("No complete, exported and logged Wan checkpoint is available")
    return max(candidates, key=lambda item: item[0])[1]


def stop_owned_tree(root_identity, *, immediate=False):
    """Use the saved UID/PID/start time and descendants, never a process-name kill."""
    if not process_alive(root_identity):
        return
    processes = snapshot()
    root = processes.get(root_identity["pid"])
    if root is None or root.started != root_identity["start_ticks"]:
        return
    owned = {root.pid: root}
    while True:
        children = {pid: p for pid, p in processes.items()
                    if pid not in owned and p.ppid in owned}
        if not children:
            break
        owned.update(children)
    signal_process(root, signal.SIGKILL if immediate else signal.SIGTERM)
    for signum, seconds in ((None, 90 if not immediate else 3),
                            (signal.SIGTERM, 15), (signal.SIGKILL, 5)):
        if signum is not None:
            for process in owned.values():
                signal_process(process, signum)
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if not any(process_alive(identity(p)) for p in owned.values()):
                return
            time.sleep(0.5)
    raise RuntimeError("An owned process did not release after handoff cleanup")


def verify(plan, *, check_code=True):
    if Path("/proc/sys/kernel/random/boot_id").read_text().strip() != plan["boot_id"]:
        raise RuntimeError("Run this queue on the recorded GPU node")
    if check_code and {name: sha256(PROJECT / "experiments" / name) for name in CODE} != plan["code"]:
        raise RuntimeError("Handoff/evaluation code changed after arming")
    for filename, digest in plan["inputs"].items():
        if sha256(Path(filename)) != digest:
            raise RuntimeError(f"Frozen handoff input changed: {filename}")
    for row in plan["baseline_assets"] + plan["metric_assets"]:
        path = Path(row["path"])
        if not path.is_file() or path.stat().st_size != row["size"] or path.stat().st_mtime_ns != row["mtime_ns"]:
            raise RuntimeError(f"Pinned model/metric asset changed: {path}")


def arm(args):
    proof = read_json(args.reshard_proof)
    if (proof.get("status") != "passed" or proof.get("source_world") != 4
            or proof.get("target_world") != 8 or proof.get("effective_batch") != 16
            or not proof.get("model_and_optimizer_restored_exactly")
            or not proof.get("eight_rank_checkpoint_saved")):
        raise ValueError("A successful four-to-eight DCP migration proof is required")
    source = read_json(args.training_plan)
    original_eval = read_json(args.previous_metrics_plan)
    verify_plan(source)
    root = Path(source["output"])
    if original_eval["training_output"] != str(root):
        raise ValueError("The training and evaluation queues do not match")
    if not process_alive(original_eval["training_identity"]):
        raise RuntimeError("The original training supervisor is no longer running")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    (output / "logs").mkdir()
    (output / "inputs").mkdir()
    evaluation = Path(original_eval["output"]) / "inputs/eval_manifest.json"
    shutil.copyfile(evaluation, output / "inputs/eval_manifest.json")
    shutil.copyfile(args.reshard_proof, output / "inputs/reshard_cpu_proof.json")
    controls = {}
    for role, script, pinned_plan in [
        ("old_mail", "watch_wm_update_email.py", args.training_plan),
        ("old_metrics", "queue_wm_video_metrics.py", args.previous_metrics_plan),
        ("cosmos", "wm_update.py", None),
    ]:
        matches = []
        for p in snapshot().values():
            if not p.argv or not Path(p.argv[0]).name.startswith("python"):
                continue
            if str(PROJECT / "experiments" / script) not in p.argv:
                continue
            if pinned_plan is not None and str(pinned_plan.resolve()) in p.argv:
                matches.append(p)
            elif role == "cosmos" and "--nproc_per_node=4" in p.argv and "cosmos" in p.argv and str(root / "train/cosmos") in p.argv:
                matches.append(p)
        if len(matches) != 1:
            raise ValueError(f"Expected one {role} process, found {len(matches)}")
        controls[role] = identity(matches[0])
    plan = {
        "schema": "cosmos-first-wan-eight-v1", "created_utc": utc(),
        "output": str(output), "source_output": str(root),
        "source_plan": str(args.training_plan.resolve()),
        "source_supervisor": original_eval["training_identity"], "controls": controls,
        "old_metrics_output": original_eval["output"],
        "boot_id": source["execution_boot_id"], "python": source["python"],
        "steps": source["steps"], "save_interval": source["save_interval"],
        "seed": source["seed"], "lr": 1e-6,
        "gpus": source["gpus"], "cosmos_gpus": source["model_allocation"]["cosmos"],
        "baseline": original_eval["baseline"],
        "baseline_assets": original_eval["baseline_assets"],
        "metric_assets": original_eval["metric_assets"],
        "manifest": str(root / "manifest.json"),
        "settings": str(root / "inputs/settings.json"),
        "training_manifest_sha256": original_eval["training_manifest_sha256"],
        "evaluation_manifest_sha256": original_eval["evaluation_manifest_sha256"],
        "evaluation_summary": original_eval["evaluation_summary"],
        "effective_batch": 16, "old_ranks": 4, "old_accumulation": 4,
        "new_ranks": 8, "new_accumulation": 2,
        "reshard_proof": proof,
        "protocol_note": original_eval["protocol_note"],
        "code": {name: sha256(PROJECT / "experiments" / name) for name in CODE},
    }
    plan["inputs"] = {str(path.resolve()): sha256(path) for path in [
        args.training_plan, args.previous_metrics_plan, Path(plan["manifest"]),
        Path(plan["settings"]), output / "inputs/eval_manifest.json",
        output / "inputs/reshard_cpu_proof.json",
    ]}
    verify(plan)
    latest_wan_checkpoint(root / "train/wan", plan["training_manifest_sha256"])
    write_json(output / "plan.json", plan)
    record(output, "prepared", next_stage="wait_cosmos")
    print(json.dumps({"plan": str(output / "plan.json"), "validation": plan["evaluation_summary"]}))


def restart(args):
    """Create a fresh audited attempt after a pre-handoff evaluation failure."""
    previous = read_json(args.plan)
    old_root = Path(previous["output"])
    if read_json(old_root / "state.json").get("status") != "failed":
        raise ValueError("Restart requires a recorded failed evaluation attempt")
    for name in ("identity.json", "notifier_identity.json", "active_child.json"):
        worker = read_json(old_root / name)
        if worker and process_alive(worker):
            raise RuntimeError(f"Previous evaluation process is still alive: {name}")
    if (old_root / "selected_wan_checkpoint.json").exists():
        raise ValueError("This restart entry only supports failures before the Wan handoff")
    verify(previous, check_code=False)
    allowed_changes = {"queue_cosmos_first_wan8.py", "wm_eval_data.py"}
    for name, expected in previous["code"].items():
        if name not in allowed_changes and sha256(PROJECT / "experiments" / name) != expected:
            raise ValueError(f"Unexpected execution code change during restart: {name}")
    verify_plan(read_json(Path(previous["source_plan"])))
    if not process_alive(previous["source_supervisor"]):
        raise RuntimeError("The original Wan training supervisor is no longer running")
    if not cosmos_ready(previous):
        raise RuntimeError("Cosmos must have a complete final checkpoint before restart")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    (output / "inputs").mkdir(parents=True)
    (output / "logs").mkdir()
    for name in ("eval_manifest.json", "reshard_cpu_proof.json"):
        shutil.copyfile(old_root / "inputs" / name, output / "inputs" / name)
    plan = dict(previous, output=str(output), created_utc=utc(),
                restart_of=str(old_root), previous_plan=str(args.plan.resolve()),
                previous_plan_sha256=sha256(args.plan),
                restart_reason="Use in-memory HDF5 decoding after verified NFS file-driver EIO",
                code={name: sha256(PROJECT / "experiments" / name) for name in CODE})
    plan["inputs"] = dict(previous["inputs"])
    for path in (args.plan, output / "inputs/eval_manifest.json", output / "inputs/reshard_cpu_proof.json"):
        plan["inputs"][str(path.resolve())] = sha256(path)
    verify(plan)
    write_json(output / "plan.json", plan)
    record(output, "prepared", next_stage="cosmos_updated_metrics", restart_of=str(old_root))
    print(json.dumps({"plan": str(output / "plan.json"), "validation": plan["evaluation_summary"]}))


def cosmos_source(plan):
    return Path(plan["source_output"]) / "train/cosmos/spatial_wmevo/cosmos_wrist_v3/train/checkpoints" / f"iter_{plan['steps']:09d}"


def cosmos_ready(plan):
    source = cosmos_source(plan)
    complete = read_json(source / "wm_update_complete.json")
    if complete:
        if complete["step"] != plan["steps"] or complete["identity"]["manifest"] != plan["training_manifest_sha256"]:
            raise ValueError("Unexpected Cosmos checkpoint step or data split")
        metric = latest_metrics(Path(plan["source_output"]) / "train/cosmos/metrics.jsonl")
        return int(metric.get("step", 0)) >= plan["steps"] and not process_alive(plan["controls"]["cosmos"])
    state = read_json(Path(plan["source_output"]) / "state.json")
    if state.get("status") == "failed" or not process_alive(plan["source_supervisor"]):
        raise RuntimeError("The original training failed before Cosmos completed")
    return False


def wait_idle(plan, devices, stage):
    root = Path(plan["output"])
    last = 0
    while True:
        usage, active = gpu_usage(), gpu_compute_pids(devices)
        if not active and all(usage.get(device, 81920) <= 1024 for device in devices):
            return
        if time.monotonic() - last >= 30:
            record(root, "waiting_for_gpus", stage=stage, used_mib=usage, compute_pids=active)
            last = time.monotonic()
        time.sleep(2)


def package_cosmos(plan):
    root = Path(plan["output"]) / "cosmos_package"
    source = cosmos_source(plan)
    target = root / source.name
    target.mkdir(parents=True)
    (target / "model").symlink_to(source / "model", target_is_directory=True)
    rows = [{"path": str(Path(source.name) / "model" / f.relative_to(source / "model")),
             "size": f.stat().st_size, "sha256": sha256(f)}
            for f in sorted((source / "model").rglob("*")) if f.is_file()]
    if not rows or not (source / "model/.metadata").is_file():
        raise ValueError("Cosmos final model files are incomplete")
    tree = hashlib.sha256("".join(f"{r['sha256']}  {r['path']}\n" for r in rows).encode()).hexdigest()
    write_json(root / "checkpoint_manifest.json", {
        "checkpoint": source.name, "run_name": "spatial_wmevo",
        "format": "pytorch-distributed-checkpoint", "file_count": len(rows),
        "total_bytes": sum(r["size"] for r in rows), "tree_sha256": tree, "files": rows,
    })
    return target


def stage(plan, name, command, env):
    from spatial_wovr_launch import stop_process

    verify(plan)
    root = Path(plan["output"])
    log = root / "logs" / f"{name}.log"
    record(root, "running", stage=name, log=str(log))
    with log.open("a") as stream:
        child = subprocess.Popen(command, cwd=PROJECT, env=env, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        try:
            process = read_process(child.pid)
            if process is not None:
                write_json(root / "active_child.json", {**identity(process), "stage": name})
            code = child.wait()
            if code:
                raise RuntimeError(f"{name} failed with exit code {code}: {log}")
        finally:
            stop_process(child)
    record(root, "stage_complete", stage=name)


def evaluate(plan, model, version, checkpoint, devices, env):
    root = Path(plan["output"])
    destination = root / "video_metrics" / model / version
    destination.mkdir(parents=True)
    shutil.copyfile(root / "inputs/eval_manifest.json", destination / "eval_manifest.json")
    variable = "WAN_SPATIAL_CHECKPOINT" if model == "wan" else "COSMOS_WRIST_CHECKPOINT"
    command = [plan["python"], str(PROJECT / "experiments/wm_eval.py"), "run",
               "--model", model, "--manifest", str(destination / "eval_manifest.json"),
               "--output", str(destination), "--gpus", ",".join(devices),
               "--steps", "5", "--seed", str(plan["seed"]), "--warmup", "3", "--workers-per-gpu", "1"]
    stage(plan, f"{model}_{version}_metrics", command,
          dict(env, CUDA_VISIBLE_DEVICES=",".join(devices), **{variable: str(checkpoint)}))


def summarize(plan, models):
    root = Path(plan["output"])
    rows, changes = [], {}
    for model in models:
        versions = {}
        for version in ("baseline", "updated"):
            data = read_json(root / "video_metrics" / model / version / f"{model}_metrics.json")
            if data.get("model") != model or data.get("manifest_sha256") != plan["evaluation_manifest_sha256"]:
                raise ValueError("Metrics must use the selected model and frozen validation episodes")
            if not all(math.isfinite(data[key]) for key in METRICS) or data["FPS"] <= 0:
                raise ValueError("All five metrics must be finite and FPS positive")
            rows.append({"version": version, "model": model, "view": data["view"],
                         "episodes": data["episodes"], "frames": data["frames"],
                         **{key: data[key] for key in METRICS}})
            versions[version] = data
        for key in ("manifest_sha256", "episodes", "frames", "protocol", "metric_backbones", "metric_resolution"):
            if versions["baseline"][key] != versions["updated"][key]:
                raise ValueError(f"Comparison protocols differ: {key}")
        changes[model] = {key: {"baseline": versions["baseline"][key], "updated": versions["updated"][key],
                               "delta": versions["updated"][key] - versions["baseline"][key]} for key in METRICS}
    name = "summary" if len(models) == 2 else f"{models[0]}_summary"
    result = {"rows": rows, "changes": changes, "protocol_note": plan["protocol_note"]}
    write_json(root / f"{name}.json", result)
    with (root / f"{name}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result


def run(plan):
    from wm_eval import configure_environment
    from wm_update import runtime

    root = Path(plan["output"])
    with (root / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "identity.json").exists():
            raise RuntimeError("Do not replay an already started handoff")
        write_json(root / "identity.json", identity(read_process(os.getpid())))

        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"Signal {signum}")
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, interrupted)
        try:
            verify(plan)
            if not plan.get("restart_of"):
                old_metrics = Path(plan["old_metrics_output"])
                old_state = read_json(old_metrics / "state.json")
                if old_state.get("status") != "waiting_for_world_model":
                    raise RuntimeError("Old evaluation queue is no longer idle; inspect before replacing")
                # Its SIGTERM handler sends a failure message. This waiting
                # process has no GPU children; replace it explicitly.
                stop_owned_tree(plan["controls"]["old_metrics"], immediate=True)
                record(old_metrics, "superseded", replacement=str(root), reason="User requested Cosmos-first metrics and Wan eight-GPU continuation")
                stop_owned_tree(plan["controls"]["old_mail"])
            write_json(Path(plan["source_output"]) / "outcome_email/transferred.json", {"replacement": str(root), "utc": utc()})
            record(root, "waiting_for_cosmos", target_step=plan["steps"])
            while not cosmos_ready(plan):
                time.sleep(2)
            verify(plan)
            wait_idle(plan, plan["cosmos_gpus"], "cosmos_metrics")
            record(root, "running", stage="package_cosmos")
            cosmos = package_cosmos(plan)
            configure_environment()
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1",
                       CUDA_DEVICE_ORDER="PCI_BUS_ID",
                       WAN_SPATIAL_CHECKPOINT=plan["baseline"]["wan"],
                       COSMOS_WRIST_CHECKPOINT=plan["baseline"]["cosmos"])
            for key in ("RAY_ADDRESS", "SPATIAL_WM_UPDATE_RUN", "SPATIAL_WM_VIDEO_METRICS_QUEUE",
                        "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
                env.pop(key, None)
            evaluate(plan, "cosmos", "updated", cosmos, plan["cosmos_gpus"], env)
            evaluate(plan, "cosmos", "baseline", plan["baseline"]["cosmos"], plan["cosmos_gpus"], env)
            result = summarize(plan, ["cosmos"])
            write_json(root / "cosmos_metrics_complete.json", {"status": "complete", "utc": utc(), "result": result})
            source = Path(plan["source_output"])
            checkpoint = latest_wan_checkpoint(source / "train/wan", plan["training_manifest_sha256"])
            selected = read_json(checkpoint / "complete.json")
            write_json(root / "selected_wan_checkpoint.json", {"path": str(checkpoint), "record": selected, "utc": utc()})
            record(root, "running", stage="handoff_wan", restored_step=selected["step"], checkpoint=str(checkpoint))
            write_json(source / "handoff_requested.json", {"replacement": str(root), "reason": "User requested eight-GPU Wan continuation", "utc": utc()})
            stop_owned_tree(plan["source_supervisor"])
            record(source, "superseded", replacement=str(root), reason="Intentional four-to-eight GPU handoff", checkpoint=str(checkpoint))
            wait_idle(plan, plan["gpus"], "wan_eight_rank_resume")
            if selected["step"] < plan["steps"]:
                training = root / "train8"
                runtime(training)
                # prepare_code_runtime updates PYTHONPATH and runtime configuration.
                env.update({key: value for key, value in os.environ.items() if key in ("PYTHONPATH", "LIBERO_CONFIG_PATH", "WM_EVAL_RUNTIME_READY")})
                command = [plan["python"], "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc_per_node=8",
                           str(PROJECT / "experiments/wm_update_wan_reshard.py"),
                           "--manifest", plan["manifest"], "--settings", plan["settings"],
                           "--output", str(training / "wan"), "--resume", str(checkpoint),
                           "--steps", str(plan["steps"]), "--save-interval", str(plan["save_interval"]),
                           "--accumulation", "2", "--lr", str(plan["lr"]), "--seed", str(plan["seed"])]
                stage(plan, "wan_train8", command, dict(env, CUDA_VISIBLE_DEVICES=",".join(plan["gpus"]), NCCL_SOCKET_IFNAME="lo"))
                checkpoint = latest_wan_checkpoint(training / "wan", plan["training_manifest_sha256"])
            if read_json(checkpoint / "complete.json")["step"] != plan["steps"]:
                raise RuntimeError("Wan exited without the final complete checkpoint")
            package = root / f"pair_{plan['steps']}"
            stage(plan, "package_pair", [plan["python"], str(PROJECT / "experiments/wm_update.py"), "package",
                  "--manifest", plan["manifest"], "--wan-checkpoint", str(checkpoint),
                  "--cosmos-checkpoint", str(cosmos_source(plan)), "--output", str(package)], env)
            write_json(root / "training_complete.json", {"status": "complete", "utc": utc(), "pair": str(package)})
            evaluate(plan, "wan", "updated", checkpoint / "runtime", plan["gpus"], env)
            evaluate(plan, "wan", "baseline", plan["baseline"]["wan"], plan["gpus"], env)
            summarize(plan, ["wan", "cosmos"])
            record(root, "complete", pair=str(package), summary=str(root / "summary.csv"))
            (root / "exit_code").write_text("0\n")
        except (Exception, KeyboardInterrupt) as exc:
            record(root, "failed", error=f"{type(exc).__name__}: {exc}")
            (root / "exit_code").write_text("1\n")
            raise


def notify(plan):
    verify(plan)
    root = Path(plan["output"])
    (root / "outcome_email").mkdir(exist_ok=True)
    with (root / "outcome_email/notifier.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        mailer = OutcomeMailer(plan, DEFAULT_CONFIG)
        write_json(root / "notifier_identity.json", identity(read_process(os.getpid())))
        while True:
            state = read_json(root / "state.json")
            worker = read_json(root / "identity.json")
            if worker and not process_alive(worker) and state.get("status") not in ("complete", "failed"):
                state = read_json(root / "state.json")
                if state.get("status") not in ("complete", "failed"):
                    state = {"status": "failed", "error": "Handoff supervisor exited without a terminal result"}
            terminal = state.get("status") in ("complete", "failed")
            for key, ready, title, summary in [
                ("primary-outcome", (root / "cosmos_metrics_complete.json").exists(), "Cosmos 视频指标评测完成", root / "cosmos_summary.csv"),
                ("fallback-outcome", terminal, "Cosmos 先评测与 Wan 八卡续训流程" + ("完成" if state.get("status") == "complete" else "失败"), root / "summary.csv"),
            ]:
                if ready and key not in mailer.events:
                    body = f"{title}\nUTC：{utc()}\n输出：{root}\n阶段：{state.get('stage', state.get('status'))}\n"
                    if state.get("error"):
                        body += f"原因：{state['error']}\n"
                    if summary.exists():
                        body += summary.read_text()
                    mailer.enqueue(key, {"subject": f"[Spatial WM] {title}", "body": body,
                                         "outcome": "complete" if key == "primary-outcome" else state["status"]})
            mailer.send_due()
            sent = terminal and "fallback-outcome" in mailer.events and all(value.get("status") == "sent" for value in mailer.deliveries.values())
            status = mailer.status()
            if sent:
                status["status"] = "notified"
            mailer.persist(root / "outcome_email/state.json", status)
            if sent:
                return
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["arm", "restart", "run", "notify", "status"])
    parser.add_argument("--training-plan", type=Path)
    parser.add_argument("--previous-metrics-plan", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--reshard-proof", type=Path)
    args = parser.parse_args()
    if args.command == "arm":
        if not args.training_plan or not args.previous_metrics_plan or not args.output or not args.reshard_proof:
            parser.error("arm requires training plan, previous metrics plan, reshard proof and a fresh output")
        arm(args)
    elif args.command == "restart":
        if not args.plan or not args.output:
            parser.error("restart requires the failed --plan and a fresh --output")
        restart(args)
    else:
        if not args.plan:
            parser.error("--plan is required")
        plan = read_json(args.plan)
        if args.command == "run":
            run(plan)
        elif args.command == "notify":
            notify(plan)
        else:
            print((Path(plan["output"]) / "state.json").read_text())


if __name__ == "__main__":
    main()
