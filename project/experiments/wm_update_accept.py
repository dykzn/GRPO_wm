"""Package paired updates and compare them on identical held-out truth videos."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from wm_update_data import atomic_json, digest, load_manifest, sha256, unchanged


def package_pair(args) -> dict:
    manifest = load_manifest(args.manifest)
    wan = args.wan_checkpoint.resolve()
    if (wan / "runtime").is_dir():
        wan = wan / "runtime"
    wan_record = json.loads((wan / "wm_update.json").read_text())
    cosmos = args.cosmos_checkpoint.resolve()
    cosmos_record = json.loads((cosmos / "wm_update_complete.json").read_text())
    if (
        wan_record["manifest_sha256"] != manifest["manifest_sha256"]
        or cosmos_record["identity"]["manifest"] != manifest["manifest_sha256"]
    ):
        raise ValueError(
            "Both models must have been trained against this exact paired manifest"
        )
    if (
        not (wan / "model-00001.safetensors").is_file()
        or not (wan / "Wan2.2_VAE.pth").is_file()
    ):
        raise ValueError("Incomplete Wan runtime package")
    if not (cosmos / "model/.metadata").is_file():
        raise ValueError("Incomplete Cosmos model checkpoint")
    if args.output.exists():
        raise FileExistsError(f"Choose a new immutable pair directory: {args.output}")
    args.output.mkdir(parents=True)
    (args.output / "wan").symlink_to(wan, target_is_directory=True)
    target = args.output / "cosmos" / cosmos.name
    (target / "model").mkdir(parents=True)
    rows = []
    for file in sorted((cosmos / "model").rglob("*")):
        if file.is_file():
            # Keep directories real: the serving identity check uses rglob,
            # which does not descend into a directory symlink.
            linked = target / "model" / file.relative_to(cosmos / "model")
            linked.parent.mkdir(parents=True, exist_ok=True)
            linked.symlink_to(file.resolve())
            rows.append(
                {
                    "path": str(
                        Path(cosmos.name) / "model" / file.relative_to(cosmos / "model")
                    ),
                    "size": file.stat().st_size,
                    "sha256": sha256(file),
                }
            )
    tree = hashlib.sha256(
        "".join(f"{r['sha256']}  {r['path']}\n" for r in rows).encode()
    ).hexdigest()
    atomic_json(
        args.output / "cosmos/checkpoint_manifest.json",
        {
            "checkpoint": cosmos.name,
            "run_name": "spatial_wmevo",
            "format": "pytorch-distributed-checkpoint",
            "file_count": len(rows),
            "total_bytes": sum(r["size"] for r in rows),
            "tree_sha256": tree,
            "files": rows,
        },
    )
    pair = {
        "schema": "spatial-wm-pair-v1",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": manifest["manifest_sha256"],
        "wan": str(args.output / "wan"),
        "cosmos": str(target),
        "wan_step": wan_record["step"],
        "cosmos_step": cosmos_record["step"],
        "wan_weights_sha256": sha256(wan / "model-00001.safetensors"),
        "cosmos_tree_sha256": tree,
        "status": "candidate; requires held-out WM evaluation and downstream real LIBERO evaluation",
    }
    atomic_json(args.output / "pair.json", pair)
    print(json.dumps(pair, indent=2), flush=True)
    return pair


def evaluation_manifest(source: Path, output: Path) -> dict:
    source = load_manifest(source)
    rows, excluded = [], []
    for row in source["episodes"]:
        if row["split"] != "val":
            continue
        unchanged(row)
        if row["horizon"] < 16:
            excluded.append(
                {
                    "trajectory_id": row["trajectory_id"],
                    "reason": "fewer than 16 future frames for video metrics",
                }
            )
            continue
        rows.append(row)
    if len(rows) < 2:
        raise ValueError(
            "Video metrics require at least two held-out episodes of at least 16 frames"
        )
    result = {
        "protocol": "local-variable-length-autoregressive-v1",
        "suite": "libero_spatial",
        "chunk": 8,
        "split_status": source["split_status"],
        "update_manifest_sha256": source["manifest_sha256"],
        "conditioning": "only real initial paired RGB; autoregressive predictions thereafter",
        "episodes": rows,
        "excluded": excluded,
        "summary": {
            "episodes": len(rows),
            "success": sum(r["success"] for r in rows),
            "future_frames": sum(r["horizon"] for r in rows),
            "min_horizon": min(r["horizon"] for r in rows),
            "max_horizon": max(r["horizon"] for r in rows),
        },
    }
    result["manifest_sha256"] = digest(result)
    if output.exists() and json.loads(output.read_text()) != result:
        raise FileExistsError("Evaluation manifest is frozen")
    atomic_json(output, result)
    return result


def evaluate(args) -> None:
    env = dict(os.environ, WM_EVAL_CAPTURE_NATIVE_PRIMARY="1")
    if args.pair:
        pair = json.loads((args.pair / "pair.json").read_text())
        source = Path(pair["manifest"])
        if load_manifest(source)["manifest_sha256"] != pair["manifest_sha256"]:
            raise ValueError("Paired training manifest changed")
        env.update(
            WAN_SPATIAL_CHECKPOINT=pair["wan"], COSMOS_WRIST_CHECKPOINT=pair["cosmos"]
        )
    else:
        source = args.manifest
        pair = {
            "status": "baseline",
            "wan": env["WAN_SPATIAL_CHECKPOINT"],
            "cosmos": env["COSMOS_WRIST_CHECKPOINT"],
        }
    command = [
        sys.executable,
        str(Path(__file__).with_name("wm_eval.py")),
        "run",
        "--manifest",
        str(args.output / "eval_manifest.json"),
        "--output",
        str(args.output),
        "--gpus",
        args.gpus,
        "--steps",
        "5",
        "--seed",
        str(args.seed),
        "--workers-per-gpu",
        "1",
    ]
    if args.dry_run:
        print(
            json.dumps(
                {"source": str(source), "pair": pair, "command": command}, indent=2
            )
        )
        return
    args.output.mkdir(parents=True, exist_ok=True)
    evaluation_manifest(source, args.output / "eval_manifest.json")
    atomic_json(args.output / "wm_update_evaluation.json", pair)
    subprocess.run(command, env=env, check=True)
    # Score only after generation/scoring workers have exited. Load this small
    # classifier on one explicitly selected GPU, with native 256px primary RGB.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus.split(",")[0].strip()
    score_outcomes(args.output)


def summarize_outcomes(rows: list[dict], *, threshold: float = 0.9) -> dict:
    from calibrate_spatial_reward import confusion

    if not rows:
        raise ValueError("No labeled outcome trajectories")
    scores = np.asarray([[r["prediction_max"], r["truth_max"]] for r in rows])
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError("Invalid success classifier probabilities")
    labels = [r["success"] for r in rows]
    return {
        "threshold": threshold,
        "predicted_video": confusion(labels, scores[:, 0], threshold),
        "real_video": confusion(labels, scores[:, 1], threshold),
        "classifier_disagreements": int(
            np.count_nonzero((scores[:, 0] >= threshold) != (scores[:, 1] >= threshold))
        ),
        "episodes": rows,
        "interpretation": "Frozen reward-classifier diagnoses against simulator episode truth; classifier errors remain possible",
    }


def score_outcomes(output: Path) -> dict:
    import torch
    from calibrate_spatial_reward import load_model

    source = Path(os.environ["DIFFSYNTH_ROOT"]) / "diffsynth/models/reward_model.py"
    checkpoint = Path(
        os.environ.get(
            "WM_UPDATE_REWARD_CHECKPOINT",
            str(
                Path(os.environ["PROJECT_ROOT"])
                / "checkpoints/wovr/RLinf-Wan-LIBERO-Spatial/resnet_rm.pth"
            ),
        )
    )
    model = load_model(source, checkpoint, "cuda:0")
    manifest = json.loads((output / "eval_manifest.json").read_text())
    rows = []
    for row in manifest["episodes"]:
        path = output / "wan" / f"{row['trajectory_id']}.npz"
        maxima = []
        with np.load(path, allow_pickle=False) as data, torch.inference_mode():
            for key in ["native_prediction", "native_truth"]:
                frames = data[key]
                if frames.shape != (row["horizon"], 256, 256, 3):
                    raise ValueError(
                        "Outcome scoring requires native 256px primary RGB without resizing"
                    )
                probabilities = []
                for start in range(0, len(frames), 32):
                    tensor = (
                        torch.from_numpy(frames[start : start + 32])
                        .to("cuda", torch.float32)
                        .permute(0, 3, 1, 2)
                    )
                    probabilities.append(
                        model.net((tensor / 127.5 - 1).clamp(-1, 1))
                        .flatten()
                        .cpu()
                        .numpy()
                    )
                maxima.append(float(np.concatenate(probabilities).max()))
        rows.append(
            {
                "trajectory_id": row["trajectory_id"],
                "task_id": row["task_id"],
                "success": bool(row["success"]),
                "prediction_max": maxima[0],
                "truth_max": maxima[1],
            }
        )
    result = summarize_outcomes(rows)
    result.update(
        manifest_sha256=manifest["manifest_sha256"],
        reward_checkpoint_sha256=sha256(checkpoint),
        reward_source_sha256=sha256(source),
        preprocessing="native 256px runtime RGB; float32 /127.5-1; no autocast or horizontal flip",
    )
    atomic_json(output / "outcome_metrics.json", result)
    return result


def compare(args) -> dict:
    if args.max_regression < 0:
        raise ValueError("max-regression must be nonnegative")
    baseline = json.loads((args.baseline / "metrics.json").read_text())
    candidate = json.loads((args.candidate / "metrics.json").read_text())
    if {r["model"] for r in baseline} != {"wan", "cosmos"} or {
        r["model"] for r in candidate
    } != {"wan", "cosmos"}:
        raise ValueError("Need complete metrics for both views and both model versions")
    records = []
    for model in ["wan", "cosmos"]:
        before = next(r for r in baseline if r["model"] == model)
        after = next(r for r in candidate if r["model"] == model)
        for key in [
            "manifest_sha256",
            "episodes",
            "frames",
            "protocol",
            "metric_backbones",
            "metric_resolution",
        ]:
            if before[key] != after[key]:
                raise ValueError(
                    f"Cannot compare different evaluation protocols or episodes: {key}"
                )
        metrics = {
            key: {
                "baseline": before[key],
                "candidate": after[key],
                "delta": after[key] - before[key],
            }
            for key in ["LPIPS", "FloLPIPS", "FID", "FVD_variable_length", "FPS"]
        }
        passed = all(
            after[key] <= before[key] * (1 + args.max_regression) + 1e-12
            for key in ["LPIPS", "FloLPIPS"]
        )
        records.append(
            {"model": model, "visual_gate_passed": passed, "metrics": metrics}
        )
    outcome_gate = None
    if (args.baseline / "outcome_metrics.json").is_file() and (
        args.candidate / "outcome_metrics.json"
    ).is_file():
        old_outcomes = json.loads((args.baseline / "outcome_metrics.json").read_text())
        new_outcomes = json.loads((args.candidate / "outcome_metrics.json").read_text())
        for key in [
            "manifest_sha256",
            "reward_checkpoint_sha256",
            "reward_source_sha256",
            "threshold",
            "preprocessing",
        ]:
            if old_outcomes[key] != new_outcomes[key]:
                raise ValueError(f"Outcome diagnostic protocols differ: {key}")
        if old_outcomes["predicted_video"]["true_failure"] > 0:
            outcome_gate = (
                new_outcomes["predicted_video"]["false_positive_rate"]
                <= old_outcomes["predicted_video"]["false_positive_rate"]
            )
    result = {
        "visual_gate_passed": all(r["visual_gate_passed"] for r in records),
        "models": records,
        "max_relative_regression": args.max_regression,
        "automatic_activation": False,
        "false_success_gate_passed": outcome_gate,
        "remaining_acceptance": [
            "Check generated success against simulator truth on held-out trajectories",
            "Run matched GRPO continuation and real Spatial 10x50 policy evaluation",
        ],
        "note": "Visual metrics alone do not establish correct dynamics or policy success; small-sample FID/FVD are diagnostic",
    }
    atomic_json(args.output / "comparison.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result
