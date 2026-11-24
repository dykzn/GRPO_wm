"""One real update/save, then restore and update again, for both Spatial WMs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

from wm_update import require_idle
from wm_update_data import atomic_json, load_manifest, sha256, training_code_identity

PROJECT = Path(__file__).resolve().parents[1]


def utc():
    return datetime.now(UTC).isoformat()


def command(plan, output, model, checkpoint=None):
    common = plan["common"]
    devices = plan["gpus"] if model == "both" else plan["allocation"][model]
    result = [
        plan["python"],
        str(PROJECT / "experiments/wm_update.py"),
        "probe" if checkpoint is None else "train",
        "--model",
        model,
        "--gpus",
        ",".join(devices),
        "--ranks-per-model",
        "4",
        "--manifest",
        common["manifest"],
        "--settings",
        common["settings"],
        "--accumulation",
        "4",
        "--lr",
        "1e-6",
        "--seed",
        "20261021",
        "--output",
        str(output),
    ]
    if checkpoint is not None:
        result += ["--steps", "2", "--save-interval", "2", "--resume", str(checkpoint)]
    return result


def checkpoints(root, step):
    return {
        "wan": root / "wan" / f"step_{step:06d}",
        "cosmos": root
        / "cosmos/spatial_wmevo/cosmos_wrist_v3/train/checkpoints"
        / f"iter_{step:09d}",
    }


def verify_checkpoints(paths, step):
    reports = {}
    for model, path in paths.items():
        marker = path / (
            "complete.json" if model == "wan" else "wm_update_complete.json"
        )
        report = json.loads(marker.read_text())
        if report["step"] != step or report["world_size"] != 4:
            raise RuntimeError(f"Unexpected {model} checkpoint completion: {marker}")
        required = (
            [path / "state/.metadata", path / "runtime/model-00001.safetensors"]
            if model == "wan"
            else [
                *(
                    path / name / ".metadata"
                    for name in ("model", "optim", "scheduler", "trainer")
                ),
                path / "master_weights.pt",
            ]
        )
        required += [path / f"rng_rank{rank}.pt" for rank in range(4)]
        for item in required:
            if not item.is_file() or item.stat().st_size == 0:
                raise RuntimeError(f"Incomplete {model} checkpoint: {item}")
        reports[model] = {"checkpoint": str(path), "step": step, "complete": True}
    return reports


def run_phase(plan, name, commands, environment, peaks):
    from spatial_wovr_launch import stop_process

    output = Path(plan["output"])
    if training_code_identity() != plan["training_code_identity"]:
        raise RuntimeError("Training code changed during the probe")
    for path, digest in plan["input_hashes"].items():
        if sha256(Path(path)) != digest:
            raise RuntimeError(f"Probe input changed: {path}")
    processes, streams = [], []
    atomic_json(
        output / "state.json",
        {"status": "running", "phase": name, "pid": os.getpid(), "updated_utc": utc()},
    )
    print(f"Phase {name} started: {utc()}", flush=True)
    try:
        for index, cmd in enumerate(commands):
            stream = (output / "logs" / f"{name}_{index}.log").open("a")
            streams.append(stream)
            processes.append(
                subprocess.Popen(
                    cmd,
                    cwd=PROJECT,
                    env=environment,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
        while True:
            codes = [p.poll() for p in processes]
            if any(code is not None and code != 0 for code in codes):
                raise RuntimeError(
                    f"Phase {name} failed: exit codes {codes}; see {output / 'logs'}"
                )
            sample = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            )
            rows = []
            for gpu, used, total, utilization in csv.reader(
                sample.stdout.splitlines(), skipinitialspace=True
            ):
                if gpu not in plan["gpus"]:
                    continue
                rows.append(
                    {
                        "gpu": gpu,
                        "used_mib": int(used),
                        "total_mib": int(total),
                        "utilization": int(utilization),
                    }
                )
                peaks[gpu] = max(peaks.get(gpu, 0), int(used))
            with (output / "gpu_usage.jsonl").open("a") as stream:
                stream.write(
                    json.dumps({"utc": utc(), "phase": name, "gpus": rows}) + "\n"
                )
            if all(code == 0 for code in codes):
                break
            time.sleep(2)
    finally:
        for process in reversed(processes):
            stop_process(process)
        for stream in streams:
            stream.close()
    print(f"Phase {name} completed: {utc()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chain-plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    source = json.loads(args.chain_plan.read_text())
    if Path("/proc/sys/kernel/random/boot_id").read_text().strip() != source["boot_id"]:
        parser.error("Run on the GPU node recorded in the chain plan")
    wm = Path(source["wm_plan"]).parent
    common = {
        "manifest": str(wm / "pilot_manifest.json"),
        "settings": str(wm / "inputs/settings.json"),
    }
    manifest = load_manifest(Path(common["manifest"]))
    output = args.output.resolve()
    plan = {
        "output": str(output),
        "python": source["python"],
        "boot_id": source["boot_id"],
        "created_utc": utc(),
        "common": common,
        "gpus": source["gpus"],
        "allocation": {"wan": source["gpus"][:4], "cosmos": source["gpus"][4:]},
        "manifest_sha256": manifest["manifest_sha256"],
        "training_code_identity": training_code_identity(),
        "input_hashes": {path: sha256(Path(path)) for path in common.values()},
    }
    fresh, resumed = output / "fresh", output / "resumed"
    first_checkpoints = checkpoints(fresh, 1)
    plan["commands"] = {
        "update_and_save": [command(plan, fresh, "both")],
        "restore_and_update": [
            command(plan, resumed, model, first_checkpoints[model])
            for model in ("wan", "cosmos")
        ],
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    require_idle(plan["gpus"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    atomic_json(output / "plan.json", plan)
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        WAN_SPATIAL_CHECKPOINT=source["baseline_wan"],
        COSMOS_WRIST_CHECKPOINT=source["baseline_cosmos"],
        SPATIAL_WM_UPDATE_PROBE=str(output),
        PYTHONUNBUFFERED="1",
    )
    for key in ("SPATIAL_FOLLOWUP_CHAIN", "SPATIAL_FOLLOWUP_ROLE", "RAY_ADDRESS"):
        env.pop(key, None)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, interrupted)
    peaks = {}
    try:
        run_phase(
            plan, "update_and_save", plan["commands"]["update_and_save"], env, peaks
        )
        first = verify_checkpoints(first_checkpoints, 1)
        atomic_json(output / "first_checkpoints.json", first)
        run_phase(
            plan,
            "restore_and_update",
            plan["commands"]["restore_and_update"],
            env,
            peaks,
        )
        second = verify_checkpoints(checkpoints(resumed, 2), 2)
        atomic_json(
            output / "result.json",
            {
                "status": "passed",
                "first": first,
                "restored": second,
                "gpu_peak_used_mib": peaks,
                "completed_utc": utc(),
            },
        )
        atomic_json(
            output / "state.json",
            {"status": "complete", "updated_utc": utc(), "pid": os.getpid()},
        )
        (output / "exit_code").write_text("0\n")
    except (Exception, KeyboardInterrupt) as exc:
        atomic_json(
            output / "state.json",
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "gpu_peak_used_mib": peaks,
                "updated_utc": utc(),
                "pid": os.getpid(),
            },
        )
        (output / "exit_code").write_text("1\n")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
