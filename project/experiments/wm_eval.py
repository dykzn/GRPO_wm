"""Prepare, generate, and score local variable-length Spatial WM rollouts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()


def configure_environment() -> None:
    """Select the existing local runtimes without changing their installation."""
    baseline = "/data41/qianb2608/wan/RLinf_opd"
    runtime = "/data73/qianb2608/grpo-world-model-runtime/venvs"
    assets = PROJECT.parent / "cosmos"
    defaults = {
        "PROJECT_ROOT": str(PROJECT), "RLINF_BASE_ROOT": baseline,
        "RLINF_ENV": runtime + "/rlinf-openpi", "COSMOS_PYTHON": runtime + "/cosmos-policy/bin/python",
        "COSMOS_BASE_ROOT": baseline + "/.tools/cosmos-policy", "LIBERO_ROOT": baseline + "/.venv/libero",
        "LIBERO_BENCHMARK_ROOT": baseline + "/.venv/libero/libero/libero",
        "OPENPI_ROOT": runtime + "/sources/openpi", "DIFFSYNTH_ROOT": runtime + "/sources/diffsynth-studio",
        "WORLDPILOT_ROOT": str(PROJECT / "third_party/WorldPilot"),
        "WAN_SPATIAL_CHECKPOINT": str(PROJECT / "checkpoints/wovr/RLinf-Wan-LIBERO-Spatial"),
        "COSMOS_WRIST_CHECKPOINT": "/data12/qianb2608/hf_models/Cosmos-WoVR-Wrist-LIBERO-Spatial-v3-step20000/iter_000020000",
        "COSMOS_VAE": str(assets / "Cosmos-Predict2-2B-Video2World/tokenizer/tokenizer.pth"),
        "COSMOS_DATASET_STATS": str(assets / "Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json"),
        "COSMOS_T5_EMBEDDINGS": str(assets / "Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl"),
        "WM_METRIC_ASSETS": str(PROJECT / "runtime_artifacts/wm_metrics"),
        "TORCH_HOME": str(PROJECT / "runtime_artifacts/wm_metrics/torch"),
        "MPLCONFIGDIR": str(PROJECT / "runtime_artifacts/wm_eval/matplotlib"),
        "COSMOS_DISABLE_CUDNN": "1", "COSMOS_DISTRIBUTED_BACKEND": "gloo",
        "COSMOS_WOVR_ACTION_DROPOUT_RATE": "0.1", "WANDB_MODE": "disabled",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "USE_TF": "0", "USE_FLAX": "0",
        "PYTHONUNBUFFERED": "1", "JAX_PLATFORMS": "cpu", "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "backend:native,expandable_segments:True",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
    paths = [str(PROJECT), str(SCRIPT.parent), str(Path(os.environ["WM_METRIC_ASSETS"]) / "packages")]
    paths += [path for path in os.environ.get("PYTHONPATH", "").split(":") if path]
    os.environ["PYTHONPATH"] = ":".join(dict.fromkeys(paths))
    sys.path[:0] = paths[:3]


def prepare_code_runtime(output: Path) -> None:
    from spatial_wovr_launch import prepare_runtime

    code = output / f"runtime_{os.getpid()}"
    code.mkdir(parents=True, exist_ok=False)
    os.environ["LIBERO_CONFIG_PATH"] = str(code / "libero_config")
    prepare_runtime(code)
    # Keep the flat camera-convention module and metric packages visible too.
    extra = [str(PROJECT), str(Path(os.environ["WM_METRIC_ASSETS"]) / "packages")]
    os.environ["PYTHONPATH"] = ":".join(extra + [os.environ["PYTHONPATH"]])
    sys.path[:0] = extra
    os.environ["WM_EVAL_RUNTIME_READY"] = "1"


def readiness(manifest: dict, selected_models=None) -> dict:
    import torch
    from wm_eval_metrics import dependencies
    from wm_eval_predict import checkpoint_identity

    models = {}
    for name in selected_models or ["wan", "cosmos"]:
        try:
            models[name] = {"ready": True, "files": checkpoint_identity(name)}
        except (ValueError, FileNotFoundError) as exc:
            models[name] = {"ready": False, "error": str(exc)}
    metrics = dependencies()
    return {"data": manifest["summary"], "models": models, "metrics": metrics,
            "visible_cuda_gpus": torch.cuda.device_count(),
            "ready": torch.cuda.is_available() and metrics["ready"]
                     and manifest["summary"]["episodes"] >= 2 and all(v["ready"] for v in models.values())}


def supervise(commands: list[tuple[list[str], dict, Path]]) -> None:
    """Run independent GPU shards, cleaning up only their owned process groups."""
    from spatial_wovr_launch import stop_process

    processes, logs = [], []
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        for command, environment, log in commands:
            stream = log.open("a")
            logs.append(stream)
            processes.append(subprocess.Popen(command, env=environment, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True))
        while True:
            codes = [process.poll() for process in processes]
            for index, code in enumerate(codes):
                if code is not None and code != 0:
                    raise RuntimeError(f"Evaluation process failed ({code}): {commands[index][2]}")
            if all(code == 0 for code in codes):
                return
            time.sleep(2)
    finally:
        for process in reversed(processes):
            stop_process(process)
        for stream in logs:
            stream.close()


def prediction_slots(gpus: str, workers_per_gpu: int) -> list[str]:
    """Assign independent, nonoverlapping trajectory shards to GPU slots."""
    devices = [part.strip() for part in gpus.split(",")]
    if any(not value.isdecimal() for value in devices) or len({int(value) for value in devices}) != len(devices):
        raise ValueError("--gpus must be distinct numeric host GPU IDs")
    if workers_per_gpu < 1:
        raise ValueError("--workers-per-gpu must be positive")
    return [str(int(device)) for device in devices for _ in range(workers_per_gpu)]


def prediction_commands(args, model: str, log_root: Path) -> list[tuple[list[str], dict, Path]]:
    slots = prediction_slots(args.gpus, args.workers_per_gpu)
    python = str(Path(os.environ["RLINF_ENV"]) / "bin/python") if model == "wan" else os.environ["COSMOS_PYTHON"]
    commands = []
    rendezvous_ports = set()
    for rank, device in enumerate(slots):
        command = [python, str(SCRIPT), "predict", "--model", model,
            "--manifest", str(args.manifest), "--output", str(args.output),
            "--shard-index", str(rank), "--num-shards", str(len(slots)),
            "--steps", str(args.steps), "--seed", str(args.seed), "--warmup", str(args.warmup)]
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=device,
                           WM_EVAL_WORKERS_PER_GPU=str(args.workers_per_gpu))
        for key in ["MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK"]:
            environment.pop(key, None)
        if model == "cosmos":
            # Each independent Cosmos instance may initialize a one-rank
            # process group; never let colocated instances share its port.
            while True:
                with socket.socket() as rendezvous:
                    rendezvous.bind(("127.0.0.1", 0))
                    port = rendezvous.getsockname()[1]
                if port not in rendezvous_ports:
                    rendezvous_ports.add(port)
                    break
            environment.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
                               RANK="0", LOCAL_RANK="0", WORLD_SIZE="1")
        commands.append((command, environment, log_root / f"{model}_shard{rank}.log"))
    return commands


def completed_predictions(manifest: dict, output: Path, model: str) -> dict[str, dict]:
    result = {}
    for row in manifest["episodes"]:
        path = output / model / f"{row['trajectory_id']}.json"
        if path.is_file() and path.with_suffix(".npz").is_file():
            result[row["trajectory_id"]] = json.loads(path.read_text())
    return result


def cosmos_probe_command(args, manifest: dict, device: str, log_root: Path):
    """Exercise one uncached real episode before starting all GPU replicas."""
    cached = completed_predictions(manifest, args.output, "cosmos")
    pending = [index for index, row in enumerate(manifest["episodes"])
               if row["trajectory_id"] not in cached]
    if not pending:
        return None
    probe = argparse.Namespace(**vars(args))
    probe.gpus, probe.workers_per_gpu = device, 1
    command, environment, _ = prediction_commands(probe, "cosmos", log_root)[0]
    command[command.index("--num-shards") + 1] = str(len(manifest["episodes"]))
    command[command.index("--shard-index") + 1] = str(pending[0])
    return command, environment, log_root / "cosmos_first_episode.log"


def cosmos_code_check() -> None:
    from cosmos_policy.experiments.robot.libero_temporal_utils import (
        assemble_temporal_raw_sequence, build_temporal_action_conditioned_batch,
        decode_temporal_future,
    )
    from cosmos_policy.models.temporal_video_layout import (
        FUTURE_PRIMARY_RAW_FRAME_INDICES, NUM_CONDITIONAL_LATENT_FRAMES,
        STATE_T, TOKENIZER_CHUNK_DURATION,
    )

    assert (STATE_T, TOKENIZER_CHUNK_DURATION, NUM_CONDITIONAL_LATENT_FRAMES) == (4, 13, 2)
    assert tuple(FUTURE_PRIMARY_RAW_FRAME_INDICES) == tuple(range(5, 13))
    assert all(callable(function) for function in [assemble_temporal_raw_sequence,
        build_temporal_action_conditioned_batch, decode_temporal_future])
    print("Cosmos temporal inference imports and 13-frame layout passed", flush=True)


def run(args, manifest: dict) -> None:
    from wm_eval_data import atomic_json

    selected_models = [args.model] if args.model else ["wan", "cosmos"]
    report = readiness(manifest, selected_models)
    if not report["ready"]:
        print(json.dumps(report, indent=2), flush=True)
        raise RuntimeError("Preflight incomplete. Install metric assets and run from the GPU terminal.")
    slots = prediction_slots(args.gpus, args.workers_per_gpu)
    devices = list(dict.fromkeys(slots))
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "preflight.json", report)
    atomic_json(args.output / "manifest.json", manifest)
    prepare_code_runtime(args.output)
    print(f"Evaluation output: {args.output}", flush=True)
    execution_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
    execution = args.output / "executions" / execution_id
    execution.mkdir(parents=True, exist_ok=False)
    print(f"Execution logs and throughput: {execution}", flush=True)
    atomic_json(execution / "plan.json", {"manifest_sha256": manifest["manifest_sha256"],
        "gpus": devices, "workers_per_gpu": args.workers_per_gpu, "prediction_gpu_slots": slots,
        "batch_per_process": 1, "note": "Concurrent per-process FPS reflects shared GPU load; phase throughput includes startup, warmup, I/O and saving."})
    with (args.output / "gpu_preflight.log").open("w") as stream:
        try:
            subprocess.run(["nvidia-smi"], stdout=stream, stderr=subprocess.STDOUT, check=False, timeout=15)
        except FileNotFoundError:
            pass
    (execution / "gpu_preflight.log").write_text((args.output / "gpu_preflight.log").read_text())
    if "cosmos" in selected_models:
        supervise([([os.environ["COSMOS_PYTHON"], str(SCRIPT), "cosmos-code-check",
                     "--output", str(args.output)], dict(os.environ, CUDA_VISIBLE_DEVICES=""),
                    execution / "cosmos_code_check.log")])
    metric_python = str(Path(os.environ["RLINF_ENV"]) / "bin/python")
    supervise([([metric_python, str(SCRIPT), "metric-smoke", "--manifest", str(args.manifest),
                  "--output", str(args.output)], dict(os.environ, CUDA_VISIBLE_DEVICES=devices[0]),
                 args.output / "metric_smoke.log")])
    for model in selected_models:
        if model == "cosmos":
            probe = cosmos_probe_command(args, manifest, devices[0], execution)
            if probe is not None:
                print("Validating Cosmos on one real episode before launching all replicas", flush=True)
                supervise([probe])
        commands = prediction_commands(args, model, execution)
        before = completed_predictions(manifest, args.output, model)
        print(f"Generating {model}: {len(before)}/{len(manifest['episodes'])} cached; "
              f"{len(commands)} processes, {args.workers_per_gpu} per GPU", flush=True)
        started = time.perf_counter()
        status = "interrupted_or_failed"
        try:
            supervise(commands)
            status = "completed"
        finally:
            elapsed = time.perf_counter() - started
            after = completed_predictions(manifest, args.output, model)
            new_ids = sorted(set(after) - set(before))
            new_frames = sum(after[key]["frames"] for key in new_ids)
            atomic_json(execution / f"{model}_throughput.json", {
                "model": model, "manifest_sha256": manifest["manifest_sha256"], "status": status,
                "gpus": devices, "workers_per_gpu": args.workers_per_gpu,
                "processes": len(commands), "cached_episodes": len(before),
                "new_episodes": len(new_ids), "new_trajectory_ids": new_ids,
                "new_frames": new_frames, "phase_wall_seconds": elapsed,
                "end_to_end_FPS_all_gpus": new_frames / elapsed,
                "includes": "model loading, warmup, prediction, data I/O, resize and compressed saving",
            })
        print(f"{model}: {new_frames} new frames / {elapsed:.1f}s = "
              f"{new_frames / elapsed:.2f} end-to-end FPS across {len(devices)} GPUs", flush=True)
    # Model processes have exited before metric backbones load on the GPU.
    command = [str(Path(os.environ["RLINF_ENV"]) / "bin/python"), str(SCRIPT), "score",
               "--manifest", str(args.manifest), "--output", str(args.output),
               "--metric-batch", str(args.metric_batch)]
    if args.model:
        command += ["--model", args.model]
    supervise([(command, dict(os.environ, CUDA_VISIBLE_DEVICES=devices[0]), args.output / "metrics.log")])
    print(f"Complete: {args.output / 'metrics.csv'}", flush=True)


def main() -> int:
    configure_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "check", "run", "predict", "score", "metric-smoke", "cosmos-code-check"])
    parser.add_argument("--data-root", type=Path, default=Path("/data41/qianb2608/wan/hf_home/LIBERO-Cosmos-Policy/all_episodes"))
    parser.add_argument("--manifest", type=Path, default=PROJECT / "runtime_artifacts/wm_eval/manifest.json")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    parser.add_argument("--output", type=Path, default=Path(f"/data73/qianb2608/wm_eval/spatial_{stamp}"))
    parser.add_argument("--limit", type=int, default=0, help="0=all complete episodes; positive=task round-robin smoke subset")
    parser.add_argument("--verify-frames", action="store_true", help="Fully decode all selected RGB while preparing; can be slow on NFS")
    parser.add_argument("--gpus", default=os.environ.get("WM_EVAL_GPUS", "0"))
    parser.add_argument("--workers-per-gpu", type=int, default=1,
                        help="Independent generation processes per GPU; start with 2 for throughput experiments")
    parser.add_argument("--model", choices=["wan", "cosmos"])
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--metric-batch", type=int, default=8)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if args.command in {"metric-smoke", "score"}:
        from wm_eval_metrics import cuda_library_environment

        environment = cuda_library_environment()
        if environment["LD_LIBRARY_PATH"] != os.environ.get("LD_LIBRARY_PATH", ""):
            # glibc reads LD_LIBRARY_PATH at process startup. Re-exec only the
            # scoring interpreter; the Wan/Cosmos runtimes retain their paths.
            os.execve(sys.executable, [sys.executable, str(SCRIPT), *sys.argv[1:]], environment)
    args.manifest, args.output = args.manifest.resolve(), args.output.resolve()
    if args.limit < 0 or args.steps < 1 or args.warmup < 1 or args.metric_batch < 1 or args.workers_per_gpu < 1:
        parser.error("limit>=0, steps>=1, warmup>=1, metric-batch>=1, workers-per-gpu>=1 required")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("shard-index must be in [0, num-shards)")
    from wm_eval_data import atomic_json, prepare_manifest

    if args.command == "cosmos-code-check":
        if not os.environ.get("WM_EVAL_RUNTIME_READY"):
            prepare_code_runtime(args.output)
        cosmos_code_check()
        return 0
    if args.command == "metric-smoke":
        from wm_eval_metrics import smoke

        smoke(args)
        return 0
    if args.command == "prepare" or not args.manifest.is_file():
        manifest = prepare_manifest(args.data_root, args.manifest, args.limit, args.verify_frames)
    else:
        manifest = json.loads(args.manifest.read_text())
        if args.limit and len(manifest["episodes"]) != args.limit:
            raise ValueError("Use a separate --manifest path for a different --limit")
    if args.command == "prepare":
        print(json.dumps(manifest["summary"], indent=2), flush=True)
        print(f"Manifest: {args.manifest}", flush=True)
    elif args.command == "check":
        report = readiness(manifest, [args.model] if args.model else None)
        path = args.manifest.parent / "readiness.json"
        atomic_json(path, report)
        print(json.dumps(report, indent=2), flush=True)
        print(f"Readiness: {path}", flush=True)
        return 0 if report["ready"] else 2
    elif args.command == "run":
        run(args, manifest)
    elif args.command == "predict":
        if not args.model:
            parser.error("predict requires --model")
        if not os.environ.get("WM_EVAL_RUNTIME_READY"):
            prepare_code_runtime(args.output)
        from wm_eval_predict import predict

        predict(args)
    else:
        from wm_eval_metrics import score

        score(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
