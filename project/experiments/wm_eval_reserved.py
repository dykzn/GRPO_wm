"""Evaluate with one reusable CUDA pool per GPU and merge metric statistics."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np

from wm_eval_data import FeatureMoments, atomic_json, frechet_distance

SCRIPT = Path(__file__).resolve()
METRICS = ("FPS", "LPIPS", "FID", "FVD_variable_length", "FloLPIPS")
MOMENTS = ("fid_real", "fid_pred", "fvd_real", "fvd_pred")


def utc():
    return datetime.now(timezone.utc).isoformat()


def state(args, status, **details):
    value = dict(status=status, updated_utc=utc(), pid=os.getpid(), **details)
    atomic_json(args.output / "state.json", value)
    with (args.output / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(value) + "\n")
    print(json.dumps(value), flush=True)


@contextmanager
def evaluation_pool(args):
    """The computing worker retains its own cache; no separate filler process."""
    import torch
    from wm_update_memory import MemorySettings, TrainingMemoryReservation, memory_hooks

    torch.cuda.set_device(0)
    device = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    output = args.output / "memory" / args.command / f"gpu_{device}" / str(os.getpid())
    pool = TrainingMemoryReservation(torch, output, MemorySettings(), args.model)
    with memory_hooks(torch, pool):
        pool.initialize()
        yield pool
        pool.record("evaluation_complete")


def merge_moments(destination, source):
    """Parallel Chan aggregation, including the between-shard covariance."""
    if source.count == 0:
        return
    if destination.count == 0:
        destination.count = source.count
        destination.mean = source.mean.copy()
        destination.m2 = source.m2.copy()
        return
    if destination.mean.shape != source.mean.shape:
        raise ValueError("Metric shard feature dimensions differ")
    total = destination.count + source.count
    delta = source.mean - destination.mean
    destination.m2 += source.m2 + np.outer(delta, delta) * (
        destination.count * source.count / total
    )
    destination.mean += delta * (source.count / total)
    destination.count = total


def prediction_records(args, manifest):
    records, identities = [], set()
    for row in manifest["episodes"]:
        path = args.output / args.model / f"{row['trajectory_id']}.npz"
        metadata_path = path.with_suffix(".json")
        if not path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"Incomplete prediction: {path}")
        metadata = json.loads(metadata_path.read_text())
        if metadata["episode"] != row or metadata["model"] != args.model:
            raise ValueError(f"Prediction does not match evaluation manifest: {path}")
        identities.add(metadata["settings_sha256"])
        records.append((row, path, metadata))
    if len(identities) != 1:
        raise ValueError("Mixed model/checkpoint/settings in evaluation predictions")
    return records, identities.pop()


def score_shard(args):
    from wm_eval_metrics import MetricModels, check_pair

    manifest = json.loads(args.manifest.read_text())
    records, identity = prediction_records(args, manifest)
    models = MetricModels()
    moments = {name: FeatureMoments() for name in MOMENTS}
    episodes = []
    lpips_sum, lpips_count, motion_sum, motion_count = 0.0, 0, 0.0, 0
    frames, seconds = 0, 0.0
    for index, (row, path, metadata) in enumerate(records[args.shard_index::args.num_shards]):
        with np.load(path, allow_pickle=False) as archive:
            predicted, truth = archive["prediction"], archive["truth"]
        length = row["horizon"]
        check_pair(predicted, truth, length)
        local_lpips, local_motion = [], []
        for start in range(0, length, args.metric_batch):
            end = min(start + args.metric_batch, length)
            values, pred_features, real_features = models.frames(predicted[start:end], truth[start:end])
            local_lpips.extend(values.tolist())
            moments["fid_pred"].add(pred_features)
            moments["fid_real"].add(real_features)
        for start in range(0, length - 1, args.metric_batch):
            end = min(start + args.metric_batch, length - 1)
            values = models.motion(predicted[start:end], truth[start:end],
                                  predicted[start+1:end+1], truth[start+1:end+1])
            local_motion.extend(values.tolist())
        moments["fvd_pred"].add(models.video(predicted))
        moments["fvd_real"].add(models.video(truth))
        lpips_sum += float(np.sum(local_lpips, dtype=np.float64))
        lpips_count += len(local_lpips)
        motion_sum += float(np.sum(local_motion, dtype=np.float64))
        motion_count += len(local_motion)
        frames += length
        seconds += metadata["seconds"]
        episodes.append(dict(trajectory_id=row["trajectory_id"], task_id=row["task_id"],
                             frames=length, LPIPS=float(np.mean(local_lpips)),
                             FloLPIPS=float(np.mean(local_motion)), FPS=metadata["fps"]))
        print(f"Metrics {args.model} shard={args.shard_index}: {index+1}, frames={length}", flush=True)
    root = args.output / "metric_shards" / args.model
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"shard_{args.shard_index:03d}.npz"
    payload = {}
    for name, moment in moments.items():
        payload.update({name + "_count": np.array(moment.count, dtype=np.int64),
                        name + "_mean": moment.mean if moment.count else np.empty(0),
                        name + "_m2": moment.m2 if moment.count else np.empty((0, 0))})
    temporary = target.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **payload)
    temporary.replace(target)
    atomic_json(target.with_suffix(".json"), dict(
        manifest_sha256=manifest["manifest_sha256"], settings_sha256=identity,
        shard_index=args.shard_index, num_shards=args.num_shards, model=args.model,
        episodes=episodes, frames=frames, generation_seconds=seconds,
        lpips_sum=lpips_sum, lpips_count=lpips_count,
        motion_sum=motion_sum, motion_count=motion_count,
    ))


def merge_scores(args, manifest, shards):
    _, identity = prediction_records(args, manifest)
    moments = {name: FeatureMoments() for name in MOMENTS}
    episodes, totals = [], dict(frames=0, generation_seconds=0.0,
                               lpips_sum=0.0, lpips_count=0, motion_sum=0.0, motion_count=0)
    for rank in range(shards):
        target = args.output / "metric_shards" / args.model / f"shard_{rank:03d}.npz"
        record = json.loads(target.with_suffix(".json").read_text())
        expected = dict(manifest_sha256=manifest["manifest_sha256"], settings_sha256=identity,
                        shard_index=rank, num_shards=shards, model=args.model)
        if any(record.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Mixed or stale metric shard: {target}")
        expected_rows = manifest["episodes"][rank::shards]
        if [row["trajectory_id"] for row in record["episodes"]] != [row["trajectory_id"] for row in expected_rows]:
            raise ValueError(f"Missing, duplicate or reordered metric episodes: {target}")
        if (record["frames"] != sum(row["horizon"] for row in expected_rows)
                or record["lpips_count"] != record["frames"]
                or record["motion_count"] != record["frames"] - len(expected_rows)):
            raise ValueError(f"Invalid frame weighting in metric shard: {target}")
        with np.load(target, allow_pickle=False) as archive:
            for name in MOMENTS:
                source = FeatureMoments()
                source.count = int(archive[name + "_count"])
                source.mean = archive[name + "_mean"].copy()
                source.m2 = archive[name + "_m2"].copy()
                count = record["frames"] if name.startswith("fid_") else len(expected_rows)
                if source.count != count:
                    raise ValueError(f"Invalid distribution sample count: {target}/{name}")
                merge_moments(moments[name], source)
        episodes.extend(record["episodes"])
        for key in totals:
            totals[key] += record[key]
    order = {row["trajectory_id"]: index for index, row in enumerate(manifest["episodes"])}
    episodes.sort(key=lambda row: order[row["trajectory_id"]])
    if len(episodes) != len(order) or len(episodes) != len(manifest["episodes"]):
        raise ValueError("Evaluation episodes are missing or duplicated")
    result = dict(model=args.model, view="wrist" if args.model == "cosmos" else "primary",
                  native_resolution=224 if args.model == "cosmos" else 256, metric_resolution=224,
                  episodes=len(episodes), frames=totals["frames"],
                  FPS=totals["frames"] / totals["generation_seconds"],
                  LPIPS=totals["lpips_sum"] / totals["lpips_count"],
                  FID=frechet_distance(moments["fid_real"], moments["fid_pred"]),
                  FVD_variable_length=frechet_distance(moments["fvd_real"], moments["fvd_pred"]),
                  FloLPIPS=totals["motion_sum"] / totals["motion_count"],
                  generation_seconds=totals["generation_seconds"],
                  macro_episode_LPIPS=float(np.mean([row["LPIPS"] for row in episodes])),
                  macro_episode_FloLPIPS=float(np.mean([row["FloLPIPS"] for row in episodes])),
                  horizon_min=min(row["frames"] for row in episodes),
                  horizon_max=max(row["frames"] for row in episodes),
                  manifest_sha256=manifest["manifest_sha256"], protocol=manifest["protocol"],
                  split_status=manifest["split_status"], metric_shards=shards,
                  aggregation="FPS=sum(frames)/sum(seconds); LPIPS per-frame; FloLPIPS per-adjacent-pair; FID/FVD from globally merged feature moments",
                  metric_backbones=dict(LPIPS="AlexNet v0.1", FID="pytorch-fid Inception-v3 2048-d",
                                        FVD="StyleGAN-V TF-Hub Kinetics-400 I3D TorchScript 400-d",
                                        FloLPIPS="official danier97/flolpips PWCNet + AlexNet v0.1"))
    atomic_json(args.output / f"{args.model}_episode_metrics.json", episodes)
    atomic_json(args.output / f"{args.model}_metrics.json", result)
    atomic_json(args.output / "metrics.json", [result])
    with (args.output / "metrics.csv").open("w", newline="") as stream:
        columns = ["model", "episodes", "frames", *METRICS]
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerow(result)
    if args.reference_comparison:
        with args.reference_comparison.open() as stream:
            rows = list(csv.DictReader(stream))
        rows.append(dict(version="reconstructed_from200_step2000", step=2000, source="current_run", **result))
        with (args.output / "comparison.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["version", "step", "source", "episodes", "frames", *METRICS], extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        atomic_json(args.output / "comparison_note.json", dict(
            historical_reference=str(args.reference_comparison),
            quality_protocol="same held-out trajectories, horizons, preprocessing and metric backbones",
            fps_note="Historical FPS requires matched hardware/runtime for direct speed comparison."))
    print(json.dumps(result, indent=2), flush=True)
    return result


def worker_command(command):
    argv, environment, log = command
    argv = list(argv)
    argv[1] = str(SCRIPT)
    environment = dict(environment, LOCAL_WORLD_SIZE="1", TORCHINDUCTOR_COMPILE_THREADS="1")
    return argv, environment, log


def run(args):
    import wm_eval
    from wm_update_memory import device_process_snapshot
    from stop_spatial_grpo import read_process

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "workflow.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.output / "worker_identity.json").exists():
            raise FileExistsError("This evaluation already ran; choose a new output")
        own = read_process(os.getpid())
        atomic_json(args.output / "worker_identity.json", dict(pid=own.pid, start_ticks=own.started))
        try:
            manifest = json.loads(args.manifest.read_text())
            devices = wm_eval.prediction_slots(args.gpus, 1)
            if len(manifest["episodes"]) < len(devices):
                raise ValueError("Require at least one evaluation episode per GPU")
            for device in devices:
                snapshot = device_process_snapshot(device, os.getpid())
                if snapshot["foreign_pids"]:
                    raise RuntimeError(f"GPU {device} is occupied: {snapshot['foreign_pids']}")
            report = wm_eval.readiness(manifest, [args.model])
            if not report["ready"]:
                raise RuntimeError(f"Evaluation preflight failed: {report}")
            atomic_json(args.output / "preflight.json", report)
            atomic_json(args.output / "manifest.json", manifest)
            atomic_json(args.output / "plan.json", dict(
                created_utc=utc(), output=str(args.output), model=args.model, gpus=devices,
                manifest=str(args.manifest), manifest_sha256=manifest["manifest_sha256"],
                summary=manifest["summary"], seed=args.seed, inference_steps=args.steps,
                warmup_chunks=args.warmup, workers_per_gpu=1, metric_shards=len(devices),
                checkpoint=os.environ.get("COSMOS_WRIST_CHECKPOINT"),
                memory_policy="computing workers retain reusable native allocator cache; headroom=3.5 GiB"))
            wm_eval.prepare_code_runtime(args.output)
            logs = args.output / "logs"
            logs.mkdir(exist_ok=True)
            state(args, "running", stage="backend_check")
            if args.model == "cosmos":
                wm_eval.supervise([([os.environ["COSMOS_PYTHON"], str(wm_eval.SCRIPT), "cosmos-code-check", "--output", str(args.output)],
                                    dict(os.environ), logs / "cosmos_code_check.log")])
            metric_python = str(Path(os.environ["RLINF_ENV"]) / "bin/python")
            smoke = [metric_python, str(SCRIPT), "metric-smoke", "--manifest", str(args.manifest), "--output", str(args.output)]
            wm_eval.supervise([(smoke, dict(os.environ, CUDA_VISIBLE_DEVICES=devices[0], LOCAL_RANK="0", RANK="0", LOCAL_WORLD_SIZE="1"), logs / "metric_smoke.log")])
            if args.model == "cosmos":
                probe = wm_eval.cosmos_probe_command(args, manifest, devices[0], logs)
                if probe:
                    state(args, "running", stage="first_episode")
                    wm_eval.supervise([worker_command(probe)])
            state(args, "running", stage="generation", gpus=devices)
            before = wm_eval.completed_predictions(manifest, args.output, args.model)
            started = time.perf_counter()
            commands = [worker_command(item) for item in wm_eval.prediction_commands(args, args.model, logs)]
            wm_eval.supervise(commands)
            elapsed = time.perf_counter() - started
            after = wm_eval.completed_predictions(manifest, args.output, args.model)
            new_frames = sum(record["frames"] for key, record in after.items() if key not in before)
            atomic_json(args.output / "throughput.json", dict(gpus=devices, new_episodes=len(after)-len(before),
                new_frames=new_frames, phase_wall_seconds=elapsed, end_to_end_FPS_all_gpus=new_frames/elapsed,
                includes="model loading, warmup, generation, I/O, resize and saving; first-episode probe excluded"))
            prediction_records(args, manifest)
            state(args, "running", stage="metrics", gpus=devices)
            commands = []
            for rank, device in enumerate(devices):
                command = [metric_python, str(SCRIPT), "score", "--model", args.model,
                           "--manifest", str(args.manifest), "--output", str(args.output),
                           "--num-shards", str(len(devices)), "--shard-index", str(rank),
                           "--metric-batch", str(args.metric_batch)]
                environment = dict(os.environ, CUDA_VISIBLE_DEVICES=device, LOCAL_RANK="0", RANK="0", LOCAL_WORLD_SIZE="1")
                commands.append((command, environment, logs / f"metrics_shard{rank}.log"))
            wm_eval.supervise(commands)
            state(args, "running", stage="merge")
            result = merge_scores(args, manifest, len(devices))
            state(args, "complete", stage="metrics", metrics=str(args.output / "metrics.csv"), scores={key:result[key] for key in METRICS})
            (args.output / "exit_code").write_text("0\n")
        except BaseException as error:
            state(args, "failed", error=f"{type(error).__name__}: {error}")
            (args.output / "exit_code").write_text("1\n")
            raise


def notify(args):
    from queue_spatial_wm_update import process_alive
    from spatial_followup_email import OutcomeMailer, read_json
    from watch_training_email import DEFAULT_CONFIG

    mailer = OutcomeMailer(dict(output=str(args.output)), DEFAULT_CONFIG)
    while True:
        result = read_json(args.output / "state.json")
        identity = read_json(args.output / "worker_identity.json")
        if identity and not process_alive(identity) and result.get("status") not in {"complete", "failed"}:
            result = read_json(args.output / "state.json")
            if result.get("status") not in {"complete", "failed"}:
                result = dict(result, status="failed", error="Evaluation supervisor exited unexpectedly")
        if result.get("status") in {"complete", "failed"}:
            title = "Cosmos 八卡五项指标评测完成" if result["status"] == "complete" else "Cosmos 八卡评测停止"
            mailer.enqueue("primary-outcome", dict(subject=f"[Cosmos] {title}", outcome=result["status"],
                           body=f"{title}\n输出：{args.output}\n结果：{json.dumps(result,ensure_ascii=False)}"))
        mailer.send_due()
        sent = mailer.deliveries.get("primary-outcome", {}).get("status") == "sent"
        mailer.persist(args.output / "outcome_email/state.json", dict(status="notified" if sent else "watching", updated_utc=utc(), deliveries=mailer.deliveries))
        if sent:
            return
        time.sleep(2)


def main():
    import wm_eval

    wm_eval.configure_environment()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:native,expandable_segments:True"
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "predict", "score", "metric-smoke", "notify"])
    parser.add_argument("--model", choices=["cosmos", "wan"], default="cosmos")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--workers-per-gpu", type=int, choices=[1], default=1)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--metric-batch", type=int, default=8)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--reference-comparison", type=Path)
    args = parser.parse_args()
    if args.command != "notify" and (args.manifest is None or not args.manifest.is_file()):
        parser.error("An existing frozen --manifest is required")
    if (min(args.steps,args.warmup,args.metric_batch,args.num_shards) < 1
            or not 0 <= args.shard_index < args.num_shards):
        parser.error("Invalid step, warmup, batch or shard arguments")
    if args.command in {"score", "metric-smoke"}:
        from wm_eval_metrics import cuda_library_environment
        environment = cuda_library_environment()
        if environment["LD_LIBRARY_PATH"] != os.environ.get("LD_LIBRARY_PATH", ""):
            os.execve(sys.executable, [sys.executable,str(SCRIPT),*sys.argv[1:]], environment)
    if args.command == "run":
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, lambda number, frame: (_ for _ in ()).throw(KeyboardInterrupt(f"Signal {number}")))
        run(args)
    elif args.command == "notify":
        notify(args)
    else:
        with evaluation_pool(args):
            if args.command == "predict":
                if not os.environ.get("WM_EVAL_RUNTIME_READY"):
                    wm_eval.prepare_code_runtime(args.output)
                from wm_eval_predict import predict
                predict(args)
            elif args.command == "metric-smoke":
                from wm_eval_metrics import smoke
                smoke(args)
            else:
                score_shard(args)


if __name__ == "__main__":
    main()
