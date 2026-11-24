"""Calibrate the Spatial success classifier on whole, labeled recordings.

Only the small released ResNet is loaded. Neither world model nor the policy
is needed. Source RGB is converted exactly as in the training environment.
Episode outcome labels cannot establish the true first successful frame.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path

import h5py
import numpy as np
from kir_frame_pool import (
    DEFAULT_GRASP_OUTPUT,
    FramePool,
    atomic_json,
    decode_rgb,
    retry_io,
    unchanged,
)

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_SOURCE = (
    Path("/data73/qianb2608/grpo-world-model-runtime/venvs/sources/diffsynth-studio")
    / "diffsynth/models/reward_model.py"
)
MODES = ("initial", "kir", "final_frame")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_episodes(rows: list[dict], seed: int) -> dict[int, str]:
    """Hold out 20 percent per task/outcome, keeping all frames together."""
    rng = np.random.default_rng(seed)
    splits = {}
    for task in sorted({row["task_id"] for row in rows}):
        for success in (False, True):
            indices = [
                row["episode_index"]
                for row in rows
                if row["task_id"] == task and row["success"] == success
            ]
            indices = rng.permutation(indices)
            count = min(len(indices) - 1, max(1, round(len(indices) * 0.2)))
            count = max(count, 0)
            splits.update(
                (int(index), "validation" if i < count else "calibration")
                for i, index in enumerate(indices)
            )
    return splits


def confusion(labels, scores, threshold: float) -> dict:
    labels = np.asarray(labels, dtype=bool)
    predicted = np.asarray(scores, dtype=np.float32) >= np.float32(threshold)
    tp = int(np.count_nonzero(predicted & labels))
    fp = int(np.count_nonzero(predicted & ~labels))
    fn = int(np.count_nonzero(~predicted & labels))
    tn = int(np.count_nonzero(~predicted & ~labels))
    recall = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    negatives = fp + tn
    if negatives:
        z2 = 1.959963984540054**2
        center = (fpr + z2 / (2 * negatives)) / (1 + z2 / negatives)
        radius = math.sqrt(fpr * (1 - fpr) / negatives + z2 / (4 * negatives**2))
        radius *= math.sqrt(z2) / (1 + z2 / negatives)
        interval = [max(0.0, center - radius), min(1.0, center + radius)]
    else:
        interval = [0.0, 1.0]
    return {
        "episodes": len(labels),
        "true_success": tp + fn,
        "true_failure": tn + fp,
        "predicted_success": tp + fp,
        "predicted_failure": tn + fn,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "recall": recall,
        "false_positive_rate": fpr,
        "fpr_wilson_95_low": interval[0],
        "fpr_wilson_95_high": interval[1],
        "precision": precision,
        "balanced_accuracy": (recall + 1 - fpr) / 2,
        "accuracy": (tp + tn) / len(labels) if len(labels) else 0.0,
    }


def candidates(scores) -> np.ndarray:
    """Include exact float32 decision boundaries, matching Torch comparisons."""
    values = np.asarray(scores, dtype=np.float32).ravel()
    above = np.nextafter(values, np.float32(np.inf))
    values = np.unique(np.r_[np.linspace(0, 1, 201), values, above].astype(np.float32))
    return values[(values > 0) & (values <= 1)]


def choose_threshold(labels, scores, max_fpr: float) -> dict:
    """Maximize recall subject to a calibration false-positive budget.

    The first column is the ordinary-start score; KIR scores are subsets of
    the same trajectory and hence cannot have a larger false-positive rate.
    The objective averages recall across the two actual training start modes.
    """
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores)
    if not labels.any() or labels.all():
        raise ValueError("Calibration needs both successful and failed episodes")
    options = []
    # A 0.005 grid avoids choosing an operating point one float32 ULP away
    # from a calibration example. The detailed sweep still has exact boundaries.
    for threshold in np.linspace(0.005, 1, 200):
        metrics = [confusion(labels, scores[:, i], float(threshold)) for i in (0, 1)]
        options.append(
            {
                "threshold": float(threshold),
                "worst_fpr": max(m["false_positive_rate"] for m in metrics),
                "mean_recall": float(np.mean([m["recall"] for m in metrics])),
                "mean_balanced_accuracy": float(
                    np.mean([m["balanced_accuracy"] for m in metrics])
                ),
            }
        )
    feasible = [row for row in options if row["worst_fpr"] <= max_fpr]
    best = max(
        feasible or options,
        key=lambda row: (
            0 if feasible else -row["worst_fpr"],
            row["mean_recall"],
            row["mean_balanced_accuracy"],
            -row["threshold"],
        ),
    )
    balanced = max(
        options,
        key=lambda row: (
            row["mean_balanced_accuracy"],
            -row["worst_fpr"],
            -row["threshold"],
        ),
    )
    return {
        "selected": best,
        "balanced_accuracy_best": balanced,
        "false_positive_budget": max_fpr,
        "budget_feasible": bool(feasible),
        "candidates": len(options),
    }


def episode_scores(probabilities, kir_start: int) -> tuple[float, float, float]:
    """Exclude reset images; keep every recorded successor, including tails."""
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if (
        probabilities.ndim != 1
        or not 0 <= kir_start < len(probabilities) - 1
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
    ):
        raise ValueError("Invalid frame probabilities or KIR start")
    return (
        float(probabilities[1:].max()),
        float(probabilities[kir_start + 1 :].max()),
        float(probabilities[-1]),
    )


def read_primary(row: dict) -> np.ndarray:
    unchanged(row)
    with h5py.File(row["path"], "r") as stream:
        frames = np.stack(
            [decode_rgb(value) for value in stream["primary_images_jpeg"]]
        )
    unchanged(row)
    if row["source_camera_space"] == "cosmos_model_flipud":
        frames = frames[:, :, ::-1]  # Cosmos source -> RLinf rot180 camera space.
    elif row["source_camera_space"] != "rlinf_runtime_rot180":
        raise ValueError(f"Unknown camera space: {row['source_camera_space']}")
    if frames.shape != (row["observations"], 256, 256, 3):
        raise ValueError(f"Unexpected primary RGB shape: {frames.shape}")
    return np.ascontiguousarray(frames)


def load_model(source: Path, checkpoint: Path, device: str):
    # Import this small module directly, without initializing the world models.
    spec = importlib.util.spec_from_file_location("spatial_reward_source", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ResnetRewModel(str(checkpoint)).eval().to(device)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(
    rows: list[dict],
    output: Path,
    provenance: dict,
    max_fpr: float,
    criterion: str = "balanced_accuracy",
) -> dict:
    labels = np.array([row["success"] for row in rows])
    scores = np.array([[row[f"{mode}_score"] for mode in MODES] for row in rows])
    calibration = np.array([row["split"] == "calibration" for row in rows])
    selection = choose_threshold(labels[calibration], scores[calibration, :2], max_fpr)
    threshold = selection[
        "balanced_accuracy_best" if criterion == "balanced_accuracy" else "selected"
    ]["threshold"]
    subsets = {
        "all": np.ones(len(rows), dtype=bool),
        "calibration": calibration,
        "validation": ~calibration,
    }
    requested = sorted(
        {
            0.5,
            0.7,
            0.8,
            0.9,
            0.95,
            0.99,
            1.0,
            threshold,
            selection["balanced_accuracy_best"]["threshold"],
        }
    )
    tables, sweep = [], []
    for subset, keep in subsets.items():
        for i, mode in enumerate(MODES):
            for value in candidates(scores):
                metrics = confusion(labels[keep], scores[keep, i], float(value))
                sweep.append(
                    dict(subset=subset, mode=mode, threshold=float(value), **metrics)
                )
            for value in requested:
                metrics = confusion(labels[keep], scores[keep, i], value)
                tables.append(
                    dict(subset=subset, mode=mode, threshold=value, **metrics)
                )
    selected = {
        subset: {
            mode: confusion(labels[keep], scores[keep, i], threshold)
            for i, mode in enumerate(MODES)
        }
        for subset, keep in subsets.items()
    }
    per_task = []
    for task in sorted({row["task_id"] for row in rows}):
        keep = np.array([row["task_id"] == task for row in rows])
        for i, mode in enumerate(MODES):
            per_task.append(
                dict(
                    task_id=task,
                    mode=mode,
                    threshold=threshold,
                    **confusion(labels[keep], scores[keep, i], threshold),
                )
            )
    write_csv(output / "episodes.csv", rows)
    write_csv(output / "threshold_comparison.csv", tables)
    write_csv(output / "threshold_sweep.csv", sweep)
    write_csv(output / "per_task.csv", per_task)
    report = {
        "provenance": provenance,
        "episodes": len(rows),
        "success": int(labels.sum()),
        "failure": int((~labels).sum()),
        "scored_frames_including_reset": sum(row["frames"] for row in rows),
        "selection": selection,
        "recommendation_criterion": criterion,
        "recommended_threshold": threshold,
        "selected_threshold_metrics": selected,
        "comparisons": tables,
        "training_override": f"env.train.success_reward_threshold={threshold:.10g}",
        "limitations": [
            (
                "Episode labels describe final outcome, not the first true success frame. "
                "Transient real successes on finally failed recordings cannot be distinguished."
            ),
            (
                "Initial/KIR scores use any future recorded frame >= threshold, exactly the "
                "single-frame success rule; reset frames are excluded and tails are retained."
            ),
            (
                "Validation holds out whole episodes from threshold selection. Independence "
                "from the released reward/world-model training data is not established."
            ),
            (
                "These are real recorded images. Generated-image reward errors require "
                "separate calibration; these numbers are not policy rollout success rates."
            ),
            "The small number of failed episodes limits confidence in low FPR estimates.",
        ],
    }
    atomic_json(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=DEFAULT_GRASP_OUTPUT)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT / "checkpoints/wovr/RLinf-Wan-LIBERO-Spatial/resnet_rm.pth",
    )
    parser.add_argument("--model-source", type=Path, default=DEFAULT_MODEL_SOURCE)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "runtime_artifacts/reward_calibration/spatial_500_v1",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-fpr", type=float, default=0.05)
    parser.add_argument(
        "--criterion",
        choices=["balanced_accuracy", "low_fpr"],
        default="balanced_accuracy",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Debug only; 0 scores the entire pool"
    )
    args = parser.parse_args()
    if (
        not 0 <= args.max_fpr <= 1
        or min(args.threads, args.batch_size) < 1
        or args.limit < 0
    ):
        parser.error("Invalid FPR, thread count, batch size or debug limit")
    import torch

    torch.set_num_threads(args.threads)
    device = (
        ("cuda:0" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    pool = FramePool(args.pool)
    pool.check_sources()
    starts = {
        int(pool.indices["start_episode"][i]): int(pool.indices["start_t"][i])
        for i in np.flatnonzero(pool.indices["start_kind"] == 1)
    }
    if set(starts) != {row["episode_index"] for row in pool.episodes}:
        raise ValueError("Every episode needs exactly one selected KIR start")
    splits = split_episodes(pool.episodes, args.seed)
    identity = {
        "schema": 1,
        "pool_sha256": pool.manifest["pool_sha256"],
        "checkpoint_sha256": sha256(args.checkpoint),
        "model_source_sha256": sha256(args.model_source),
        "camera_conversion": "cosmos_flipud_to_runtime_rot180_horizontal_flip_once",
        "reward_flip_lr": False,
        "normalization": "RGB uint8 -> float32 / 127.5 - 1; clamp[-1,1]",
        "success_rule": "probability >= threshold; success_consecutive_frames=1",
        "precision": "float32, no autocast",
    }
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "cache_identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Output cache belongs to a different model/pool/preprocessing")
    atomic_json(identity_path, identity)
    cache = output / "probabilities"
    cache.mkdir(exist_ok=True)
    model = load_model(args.model_source, args.checkpoint, device)
    rows, started, frame_count = [], time.monotonic(), 0
    episodes = pool.episodes[: args.limit] if args.limit else pool.episodes
    print(f"Scoring {len(episodes)} episodes on {device}; output: {output}", flush=True)
    for number, row in enumerate(episodes, 1):
        target = cache / f"episode_{row['episode_index']:04d}.npy"
        if target.exists():
            probabilities = np.load(target, allow_pickle=False)
        else:
            frames = retry_io(read_primary, row)
            chunks = []
            with torch.inference_mode():
                for offset in range(0, len(frames), args.batch_size):
                    tensor = torch.from_numpy(frames[offset : offset + args.batch_size])
                    tensor = tensor.to(device=device, dtype=torch.float32).permute(
                        0, 3, 1, 2
                    )
                    tensor = (tensor / 127.5 - 1).clamp(-1, 1)
                    chunks.append(model.net(tensor).flatten().cpu().numpy())
            probabilities = np.concatenate(chunks)
            temporary = target.with_suffix(".partial.npy")
            np.save(temporary, probabilities, allow_pickle=False)
            temporary.replace(target)
        if probabilities.shape != (row["observations"],):
            raise ValueError(f"Invalid cached score shape: {target}")
        start = starts[row["episode_index"]]
        scores = episode_scores(probabilities, start)
        rows.append(
            dict(
                episode_index=row["episode_index"],
                task_id=row["task_id"],
                episode_number=row["episode_number"],
                success=row["success"],
                split=splits[row["episode_index"]],
                frames=len(probabilities),
                kir_start=start,
                **dict(zip((f"{mode}_score" for mode in MODES), scores)),
                initial_argmax_frame=int(np.argmax(probabilities[1:])) + 1,
                kir_argmax_frame=int(np.argmax(probabilities[start + 1 :])) + start + 1,
                source=row["path"],
            )
        )
        frame_count += len(probabilities)
        elapsed = time.monotonic() - started
        if number % 10 == 0 or number == len(episodes):
            progress = {
                "episodes": number,
                "total": len(episodes),
                "frames": frame_count,
                "seconds": elapsed,
                "mean_frames_per_second": frame_count / elapsed,
                "estimated_seconds_remaining": elapsed
                / number
                * (len(episodes) - number),
            }
            atomic_json(output / "progress.json", progress)
            print(
                f"{number}/{len(episodes)} episodes | {frame_count} frames | "
                f"{frame_count / elapsed:.1f} fps | ETA {progress['estimated_seconds_remaining'] / 60:.1f} min",
                flush=True,
            )
    provenance = dict(
        **identity,
        device=device,
        torch_version=torch.__version__,
        threads=args.threads,
        batch_size=args.batch_size,
        seed=args.seed,
        checkpoint=str(args.checkpoint.resolve()),
        model_source=str(args.model_source),
        seconds=time.monotonic() - started,
        debug_limit=args.limit,
    )
    report = analyze(rows, output, provenance, args.max_fpr, args.criterion)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "episodes",
                    "success",
                    "failure",
                    "selection",
                    "selected_threshold_metrics",
                    "training_override",
                )
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
