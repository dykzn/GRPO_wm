"""After a successful WM update, compare original/new models on held-out videos."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from queue_spatial_wm_update import (
    gpu_compute_pids,
    gpu_usage,
    process_alive,
    record,
    sha256,
    utc,
    write_json,
)
from spatial_followup_email import OutcomeMailer, read_json
from stop_spatial_grpo import read_process
from watch_training_email import DEFAULT_CONFIG

PROJECT = Path(__file__).resolve().parents[1]
CODE = [
    "queue_wm_video_metrics.py",
    "wm_update.py",
    "wm_update_accept.py",
    "wm_eval.py",
    "wm_eval_data.py",
    "wm_eval_predict.py",
    "wm_eval_metrics.py",
    "spatial_wovr_launch.py",
    "world_model_grpo/cosmos_wrist_adapter.py",
]
METRICS = ("FPS", "LPIPS", "FID", "FVD_variable_length", "FloLPIPS")


def commands(plan):
    root = Path(plan["output"])
    base = [plan["python"], str(PROJECT / "experiments/wm_update.py")]
    common = ["--gpus", ",".join(plan["gpus"]), "--seed", str(plan["seed"])]
    return [
        (
            "baseline",
            base
            + [
                "evaluate",
                "--manifest",
                str(root / "inputs/training_manifest.json"),
                "--output",
                str(root / "baseline"),
            ]
            + common,
        ),
        (
            "updated",
            base
            + ["evaluate", "--pair", plan["pair"], "--output", str(root / "updated")]
            + common,
        ),
        (
            "compare",
            base
            + [
                "compare",
                "--baseline",
                str(root / "baseline"),
                "--candidate",
                str(root / "updated"),
                "--output",
                str(root / "comparison"),
            ],
        ),
    ]


def verify(plan):
    if Path("/proc/sys/kernel/random/boot_id").read_text().strip() != plan["boot_id"]:
        raise RuntimeError("Evaluation queue belongs to a different execution node")
    if plan["code"] != {name: sha256(PROJECT / "experiments" / name) for name in CODE}:
        raise RuntimeError(
            "Evaluation code changed after arming; review before rerunning"
        )
    if sha256(Path(plan["training_plan"])) != plan["training_plan_sha256"]:
        raise RuntimeError("Source training plan changed")
    for name, digest in plan["input_hashes"].items():
        if sha256(Path(plan["output"]) / "inputs" / name) != digest:
            raise RuntimeError(f"Frozen evaluation input changed: {name}")
    for row in plan["baseline_assets"] + plan["metric_assets"]:
        p = Path(row["path"])
        if (
            not p.is_file()
            or p.stat().st_size != row["size"]
            or p.stat().st_mtime_ns != row["mtime_ns"]
        ):
            raise RuntimeError(f"Evaluation asset changed: {p}")


def upstream_ready(plan):
    root = Path(plan["training_output"])
    state = read_json(root / "state.json")
    exit_path = root / "exit_code"
    if state.get("status") == "failed" or (
        exit_path.exists() and exit_path.read_text().strip() != "0"
    ):
        raise RuntimeError(
            f"World-model update did not succeed: {state.get('error', root)}"
        )
    identity = plan["training_identity"]
    if state.get("status") != "complete":
        if not process_alive(identity):
            # Cover a final success written between state and liveness reads.
            state = read_json(root / "state.json")
            if state.get("status") != "complete":
                raise RuntimeError(
                    "World-model supervisor exited before successful completion"
                )
        else:
            return False
    if not exit_path.exists() or process_alive(identity):
        return False
    pair = json.loads((Path(plan["pair"]) / "pair.json").read_text())
    if (
        pair["wan_step"] != plan["target_step"]
        or pair["cosmos_step"] != plan["target_step"]
    ):
        raise RuntimeError("Packaged world models have unexpected update steps")
    if pair["manifest_sha256"] != plan["training_manifest_sha256"]:
        raise RuntimeError("Packaged models use a different training/validation split")
    return True


def arm(args):
    from wm_eval import configure_environment
    from wm_eval_metrics import HUB_WEIGHTS, dependencies
    from wm_eval_predict import checkpoint_identity
    from wm_update_accept import evaluation_manifest
    from wm_update_data import load_manifest

    configure_environment()
    source = json.loads(args.training_plan.read_text())
    if (
        Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        != source["execution_boot_id"]
    ):
        raise RuntimeError("Arm the evaluation on the formal training node")
    status = dependencies()
    if not status["ready"]:
        raise RuntimeError(f"Video metric assets are incomplete: {status['missing']}")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a fresh evaluation output: {output}")
    output.mkdir(parents=True)
    (output / "inputs").mkdir()
    (output / "logs").mkdir()
    training_root = Path(source["output"])
    for filename, path in {
        "training_manifest.json": training_root / "manifest.json",
        "training_plan.json": args.training_plan,
    }.items():
        shutil.copyfile(path, output / "inputs" / filename)
    full = load_manifest(output / "inputs/training_manifest.json")
    evaluation = evaluation_manifest(
        output / "inputs/training_manifest.json", output / "inputs/eval_manifest.json"
    )
    baselines = {}
    for model in ("wan", "cosmos"):
        identity = read_json(training_root / "train" / model / "training_identity.json")
        baselines[model] = identity["source"]
    os.environ.update(
        WAN_SPATIAL_CHECKPOINT=baselines["wan"],
        COSMOS_WRIST_CHECKPOINT=baselines["cosmos"],
    )
    baseline_assets = checkpoint_identity("wan") + checkpoint_identity("cosmos")
    assets_root = Path(os.environ["WM_METRIC_ASSETS"])
    assets = [
        assets_root / "i3d_torchscript.pt",
        assets_root / "flolpips/flolpips.py",
        assets_root / "flolpips/pwcnet.py",
        assets_root / "flolpips/weights/v0.1/alex.pth",
    ]
    assets += [
        Path(os.environ["TORCH_HOME"]) / "hub/checkpoints" / name
        for name in HUB_WEIGHTS
    ]
    plan = {
        "schema": "spatial-wm-post-training-video-metrics-v1",
        "created_utc": utc(),
        "output": str(output),
        "training_plan": str(args.training_plan.resolve()),
        "training_plan_sha256": sha256(args.training_plan),
        "training_output": str(training_root),
        "training_identity": read_json(training_root / "worker_identity.json"),
        "boot_id": source["execution_boot_id"],
        "python": source["python"],
        "target_step": source["steps"],
        "pair": str(training_root / f"pair_{source['steps']}"),
        "gpus": source["gpus"],
        "seed": source["seed"],
        "baseline": baselines,
        "training_manifest_sha256": full["manifest_sha256"],
        "evaluation_manifest_sha256": evaluation["manifest_sha256"],
        "evaluation_summary": evaluation["summary"],
        "inference_steps": 5,
        "warmup_chunks": 3,
        "batch": 1,
        "workers_per_gpu": 1,
        "baseline_assets": baseline_assets,
        "metric_assets": [
            {
                "path": str(p.resolve()),
                "size": p.stat().st_size,
                "mtime_ns": p.stat().st_mtime_ns,
            }
            for p in assets
        ],
        "code": {name: sha256(PROJECT / "experiments" / name) for name in CODE},
        "input_hashes": {
            name: sha256(output / "inputs" / name)
            for name in (
                "training_manifest.json",
                "training_plan.json",
                "eval_manifest.json",
            )
        },
        "protocol_note": "Matched held-out episodes, actions and variable lengths aligned to full 8-frame chunks; FVD uses one 400-d I3D feature per video. Original/new comparisons are within each camera. Previous 500-video scores use different episodes and are not used as the matched baseline.",
    }
    if not plan["training_identity"]:
        raise RuntimeError("Source training process has no recorded identity")
    write_json(output / "plan.json", plan)
    verify(plan)
    record(output, "prepared", validation=evaluation["summary"])
    print(json.dumps({"output": str(output), "commands": commands(plan)}, indent=2))


def run_stage(plan, name, cmd, environment):
    from spatial_wovr_launch import stop_process

    verify(plan)
    root = Path(plan["output"])
    record(root, "running", stage=name, log=str(root / "logs" / f"{name}.log"))
    with (root / "logs" / f"{name}.log").open("a") as stream:
        child = subprocess.Popen(
            cmd,
            cwd=PROJECT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = child.wait()
            if code:
                raise RuntimeError(
                    f"Video metric stage {name} exited {code}: {root / 'logs' / (name + '.log')}"
                )
        finally:
            stop_process(child)
    record(root, "stage_complete", stage=name)


def write_summary(plan):
    root = Path(plan["output"])
    rows = []
    for version in ("baseline", "updated"):
        results = json.loads((root / version / "metrics.json").read_text())
        if {r["model"] for r in results} != {"wan", "cosmos"}:
            raise ValueError("Both camera metrics are required for each model version")
        for result in results:
            if result["manifest_sha256"] != plan["evaluation_manifest_sha256"]:
                raise ValueError("Evaluation did not use the frozen held-out manifest")
            if not all(math.isfinite(result[key]) for key in METRICS):
                raise ValueError("Video metric results must all be finite")
            rows.append(
                {
                    "version": version,
                    "model": result["model"],
                    "view": result["view"],
                    "episodes": result["episodes"],
                    "frames": result["frames"],
                    **{key: result[key] for key in METRICS},
                }
            )
    write_json(
        root / "summary.json",
        {
            "rows": rows,
            "comparison": json.loads((root / "comparison/comparison.json").read_text()),
            "protocol_note": plan["protocol_note"],
        },
    )
    with (root / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def email_result(plan, status, error=None):
    root = Path(plan["output"])
    mailer = OutcomeMailer(plan, DEFAULT_CONFIG)
    label = "完成" if status == "complete" else "未完成"
    body = f"Spatial 更新前后视频指标评测：{label}\n输出：{root}\n验证轨迹：{plan['evaluation_summary']['episodes']} 条\n指标：FPS、LPIPS、FID、变长 FVD、FloLPIPS\n"
    if error:
        body += f"原因：{error}\n"
    if (root / "summary.csv").exists():
        body += (root / "summary.csv").read_text()
    mailer.enqueue(
        "primary-outcome",
        {
            "role": "wm_video_metrics",
            "outcome": status,
            "subject": f"[Spatial WM] 更新前后视频指标评测{label}",
            "body": body,
        },
    )
    while mailer.deliveries["primary-outcome"].get("status") != "sent":
        mailer.send_due()
        mailer.persist(root / "email_state.json", mailer.deliveries["primary-outcome"])
        if mailer.deliveries["primary-outcome"].get("status") != "sent":
            time.sleep(2)


def run(plan):
    from wm_eval import configure_environment

    root = Path(plan["output"])
    with (root / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "identity.json").exists():
            raise RuntimeError(
                "This evaluation queue already ran; inspect before replaying it"
            )
        process = read_process(os.getpid())
        write_json(
            root / "identity.json", {"pid": process.pid, "start_ticks": process.started}
        )

        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"Signal {signum}")

        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, interrupted)
        try:
            verify(plan)
            record(root, "waiting_for_world_model", target_step=plan["target_step"])
            while not upstream_ready(plan):
                time.sleep(2)
            verify(plan)
            write_json(
                root / "inputs/selected_pair.json",
                json.loads((Path(plan["pair"]) / "pair.json").read_text()),
            )
            last = float("-inf")
            while True:
                usage, active = gpu_usage(), gpu_compute_pids(plan["gpus"])
                if not active and all(
                    usage.get(g, 81920) <= 1024 for g in plan["gpus"]
                ):
                    break
                if time.monotonic() - last >= 30:
                    record(
                        root, "waiting_for_gpus", used_mib=usage, compute_pids=active
                    )
                    last = time.monotonic()
                time.sleep(2)
            configure_environment()
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES="",
                PYTHONUNBUFFERED="1",
                WAN_SPATIAL_CHECKPOINT=plan["baseline"]["wan"],
                COSMOS_WRIST_CHECKPOINT=plan["baseline"]["cosmos"],
                SPATIAL_WM_VIDEO_METRICS_RUN=str(root),
            )
            for key in (
                "RAY_ADDRESS",
                "SPATIAL_WM_UPDATE_RUN",
                "PYTORCH_CUDA_ALLOC_CONF",
                "PYTORCH_ALLOC_CONF",
            ):
                env.pop(key, None)
            for name, cmd in commands(plan):
                run_stage(plan, name, cmd, env)
            rows = write_summary(plan)
            record(root, "complete", rows=rows, summary=str(root / "summary.csv"))
            (root / "exit_code").write_text("0\n")
        except (Exception, KeyboardInterrupt) as exc:
            error = f"{type(exc).__name__}: {exc}"
            record(root, "failed", error=error)
            (root / "exit_code").write_text("1\n")
            email_result(plan, "failed", error)
            raise
        email_result(plan, "complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("arm", "run", "status"))
    parser.add_argument("--training-plan", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plan", type=Path)
    args = parser.parse_args()
    if args.command == "arm":
        if not args.training_plan or not args.output:
            parser.error("arm requires --training-plan and --output")
        arm(args)
    else:
        if not args.plan:
            parser.error("A recorded --plan is required")
        plan = json.loads(args.plan.read_text())
        if args.command == "run":
            run(plan)
        else:
            print((Path(plan["output"]) / "state.json").read_text())


if __name__ == "__main__":
    main()
