#!/usr/bin/env python3
"""Train a Task- & Class-Balanced TaskEmbedResnetRewModel for LIBERO-Spatial.

Why this exists
---------------
Calibration of the released ``resnet_rm.pth`` on 500 labeled Spatial trajectories
showed severe per-task imbalance at the global ``0.9`` threshold:
- High false-positive tasks: Task 0 (40%), Task 5 (33%), Task 6 (33%), Task 7 (21%).
- High false-negative tasks: Task 4 (Recall 78.9%), Task 8 (Recall 82.6%).

This trainer addresses both root causes on primary-camera images only (no wrist view):
1. **Task conditioning**: uses ``TaskEmbedResnetRewModel`` (ResNet-18 visual backbone
   warm-started from ``resnet_rm.pth`` + ``nn.Embedding(10, 64)`` + fusion MLP).
2. **Task- & class-balanced sampling with hard-negative mining**: every epoch draws
   equal samples from all 10 tasks; within each task, 50% positive (success-episode
   tail), 30% hard negative (failed episodes, up-weighting late frames and frames
   where baseline ``resnet_rm.pth`` produced false-positive ``prob >= 0.5``), and
   20% pre-success negative (early/mid frames of successful episodes).
3. **Zero-leakage episode evaluation**: evaluates whole episodes on the exact same
   400/100 calibration/validation split and single-frame trigger rule as
   ``calibrate_spatial_reward.py``, reporting head-to-head metrics vs baseline.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
import time
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EXPERIMENTS = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENTS.parents[1]

DEFAULT_POOL = REPO_ROOT / "config/kir_pool"
DEFAULT_BASELINE_RM = REPO_ROOT / "assets/wan_baseline/resnet_rm.pth"
DEFAULT_MODEL_SOURCE = REPO_ROOT / "sources/diffsynth/diffsynth/models/reward_model.py"
DEFAULT_CALIBRATION_DIR = REPO_ROOT / "runs/resnet_libero_cosmos_spatial_500"
DEFAULT_OUTPUT = REPO_ROOT / "runs/task_balanced_rm_spatial_v1"


def load_reward_module(model_source: Path):
    spec = importlib.util.spec_from_file_location("spatial_tb_reward_source", model_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def warm_start_task_embed_from_resnet(
    task_model: nn.Module, baseline_ckpt_path: Path
) -> int:
    """Copy pretrained ResNet-18 backbone weights from ``ResnetRewModel`` into ``visual_encoder``."""
    raw = torch.load(baseline_ckpt_path, map_location="cpu", weights_only=False)
    sd = raw.get("model_state_dict", raw)
    target_sd = task_model.state_dict()
    copied = 0
    for key, val in sd.items():
        if key.startswith("net."):
            mapped = "visual_encoder." + key[len("net.") :]
            if mapped in target_sd and tuple(target_sd[mapped].shape) == tuple(val.shape):
                target_sd[mapped] = val.clone()
                copied += 1
    if copied < 100:
        raise RuntimeError(
            f"Expected >=100 ResNet backbone tensors when warm-starting; copied only {copied}"
        )
    task_model.load_state_dict(target_sd, strict=True)
    return copied


def set_trainable_layers(task_model: nn.Module, *, unfreeze_from_block: int = 4) -> None:
    """Freeze early generic conv blocks ``b1..b{k-1}`` and unfreeze late blocks + head.

    ``visual_encoder`` Sequential indices:
      0: b1, 1: b2, 2: b3, 3: b4, 4: b5, 5: AvgPool, 6: Flatten
    ``unfreeze_from_block=4`` unfreezes ``b4`` (idx 3), ``b5`` (idx 4),
    ``task_embedding``, and ``fusion_layer``.
    ``unfreeze_from_block=1`` unfreezes the entire backbone.
    """
    min_idx = max(0, int(unfreeze_from_block) - 1)
    for param in task_model.parameters():
        param.requires_grad = False
    for idx, child in enumerate(task_model.visual_encoder):
        if idx >= min_idx:
            for param in child.parameters():
                param.requires_grad = True
    for param in task_model.task_embedding.parameters():
        param.requires_grad = True
    for param in task_model.fusion_layer.parameters():
        param.requires_grad = True


def _decode_primary_frames_and_description(
    path: Path, camera_space: str
) -> tuple[np.ndarray, str]:
    from kir_frame_pool import decode_rgb

    with h5py.File(path, "r") as stream:
        frames = np.stack([decode_rgb(val) for val in stream["primary_images_jpeg"]])
        raw_desc = stream.attrs["task_description"]
        if isinstance(raw_desc, bytes):
            desc = raw_desc.decode("utf-8").strip()
        elif hasattr(raw_desc, "item"):
            val = raw_desc.item()
            desc = (val.decode("utf-8") if isinstance(val, bytes) else str(val)).strip()
        else:
            desc = str(raw_desc).strip()
    if camera_space == "cosmos_model_flipud":
        frames = frames[:, :, ::-1]
    elif camera_space != "rlinf_runtime_rot180":
        raise ValueError(f"Unknown camera space: {camera_space}")
    return np.ascontiguousarray(frames), desc


def _decode_primary_frames(path: Path, camera_space: str) -> np.ndarray:
    frames, _ = _decode_primary_frames_and_description(path, camera_space)
    return frames


def _augment_batch_tensor(x: torch.Tensor) -> torch.Tensor:
    """Geometry-preserving visual augmentation (NO horizontal flip for LIBERO-Spatial!).

    Input ``x`` is float32 in ``[-1, 1]`` with shape ``[B, 3, H, W]``.
    Simulates mild lighting jitter, tiny pixel shift (±4px), and slight Wan softness.
    """
    bsz, _, h, w = x.shape
    # 1. Brightness & contrast jitter (±12%).
    contrast = torch.empty((bsz, 1, 1, 1), device=x.device).uniform_(0.88, 1.12)
    brightness = torch.empty((bsz, 1, 1, 1), device=x.device).uniform_(-0.10, 0.10)
    x = (x * contrast + brightness).clamp(-1.0, 1.0)

    # 2. Tiny random translation (±4 px) via zero-copy roll + edge pad simulation.
    pad = 4
    padded = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    out = torch.empty_like(x)
    dy = torch.randint(0, 2 * pad + 1, (bsz,), device=x.device).tolist()
    dx = torch.randint(0, 2 * pad + 1, (bsz,), device=x.device).tolist()
    for i in range(bsz):
        out[i] = padded[i, :, dy[i] : dy[i] + h, dx[i] : dx[i] + w]

    # 3. Mild Gaussian noise (p=0.3) to match diffusion softness.
    noise_mask = (torch.rand((bsz, 1, 1, 1), device=x.device) < 0.3).float()
    out = (out + noise_mask * torch.randn_like(out) * 0.03).clamp(-1.0, 1.0)
    return out


def build_task_balanced_index(
    episodes: list[dict[str, Any]],
    *,
    baseline_probs_dir: Path | None,
    pos_tail_frames: int = 5,
    pre_success_margin: int = 20,
    hard_neg_fp_weight: float = 5.0,
    hard_neg_late_weight: float = 3.0,
) -> dict[int, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Build per-task sampling pools ``(ep_ids, frame_ids, weights)`` for train episodes.

    Strictly uses only the provided training/calibration episodes.
    Returns ``pools[task_id][bucket] = (ep_indices, frame_indices, normalized_weights)``
    where ``bucket`` is one of ``{"pos", "hard_neg", "pre_neg"}``.
    """
    pools: dict[int, dict[str, list[tuple[int, int, float]]]] = {
        t: {"pos": [], "hard_neg": [], "pre_neg": []} for t in range(10)
    }
    for local_idx, ep in enumerate(episodes):
        task_id = int(ep.get("model_task_id", ep["task_id"]))
        n_frames = int(ep["frames"])
        success = bool(ep["success"])
        base_probs = None
        if baseline_probs_dir is not None:
            pfile = baseline_probs_dir / f"episode_{int(ep['episode_index']):04d}.npy"
            if pfile.is_file():
                base_probs = np.load(pfile, allow_pickle=False)

        if success:
            pos_start = max(1, n_frames - pos_tail_frames)
            for f_idx in range(pos_start, n_frames):
                w = 1.0
                if base_probs is not None and f_idx < len(base_probs) and base_probs[f_idx] < 0.9:
                    # Up-weight false-negative success tails (e.g. Task 4).
                    w = 3.0
                pools[task_id]["pos"].append((local_idx, f_idx, w))
            pre_end = max(1, n_frames - pre_success_margin)
            for f_idx in range(0, pre_end):
                w = 1.0
                if base_probs is not None and f_idx < len(base_probs) and base_probs[f_idx] >= 0.5:
                    w = hard_neg_fp_weight
                pools[task_id]["pre_neg"].append((local_idx, f_idx, w))
        else:
            late_start = max(1, int(0.7 * n_frames))
            for f_idx in range(1, n_frames):
                w = hard_neg_late_weight if f_idx >= late_start else 1.0
                if base_probs is not None and f_idx < len(base_probs) and base_probs[f_idx] >= 0.5:
                    w = max(w, hard_neg_fp_weight)
                pools[task_id]["hard_neg"].append((local_idx, f_idx, w))

    packed: dict[int, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for task_id, buckets in pools.items():
        packed[task_id] = {}
        for bucket_name, items in buckets.items():
            if not items:
                packed[task_id][bucket_name] = (
                    np.zeros(0, dtype=np.int64),
                    np.zeros(0, dtype=np.int64),
                    np.zeros(0, dtype=np.float64),
                )
                continue
            ep_arr = np.array([x[0] for x in items], dtype=np.int64)
            fr_arr = np.array([x[1] for x in items], dtype=np.int64)
            w_arr = np.array([x[2] for x in items], dtype=np.float64)
            w_arr = w_arr / w_arr.sum()
            packed[task_id][bucket_name] = (ep_arr, fr_arr, w_arr)
    return packed


def sample_balanced_epoch(
    pools: dict[int, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]],
    *,
    samples_per_task: int,
    pos_frac: float = 0.50,
    hard_neg_frac: float = 0.30,
    rng: np.random.Generator,
) -> list[tuple[int, int, int, float]]:
    """Return shuffled list of ``(local_ep_idx, frame_idx, task_id, label)`` with exact task balance."""
    n_pos = int(round(samples_per_task * pos_frac))
    n_hard = int(round(samples_per_task * hard_neg_frac))
    n_pre = samples_per_task - n_pos - n_hard
    records: list[tuple[int, int, int, float]] = []
    for task_id in sorted(pools.keys()):
        tb = pools[task_id]
        for bucket_name, count, label in (
            ("pos", n_pos, 1.0),
            ("hard_neg", n_hard, 0.0),
            ("pre_neg", n_pre, 0.0),
        ):
            ep_arr, fr_arr, w_arr = tb[bucket_name]
            if len(ep_arr) == 0 or count <= 0:
                continue
            chosen = rng.choice(len(ep_arr), size=count, replace=True, p=w_arr)
            for idx in chosen:
                records.append((int(ep_arr[idx]), int(fr_arr[idx]), task_id, label))
    rng.shuffle(records)
    return records


def forward_task_embed_probs(
    model: nn.Module, frames_tensor: torch.Tensor, task_ids: torch.Tensor
) -> torch.Tensor:
    """Compute continuous probabilities ``[B]`` without hard rounding."""
    vis = model.visual_encoder(frames_tensor.clamp(-1.0, 1.0))
    emb = model.task_embedding(task_ids)
    return model.fusion_layer(torch.cat([vis, emb], dim=1)).squeeze(-1)


def score_episodes(
    model: nn.Module,
    episodes: list[dict[str, Any]],
    frames_by_ep: list[np.ndarray],
    *,
    device: torch.device,
    batch_size: int = 64,
) -> list[np.ndarray]:
    """Score every frame of every episode in eval mode; returns list of 1D float32 arrays."""
    model.eval()
    all_probs: list[np.ndarray] = []
    with torch.inference_mode():
        for ep, frames in zip(episodes, frames_by_ep, strict=True):
            task_id = int(ep.get("model_task_id", ep["task_id"]))
            chunks: list[np.ndarray] = []
            for offset in range(0, len(frames), batch_size):
                batch_np = frames[offset : offset + batch_size]
                x = torch.from_numpy(batch_np).to(device=device, dtype=torch.float32)
                x = (x.permute(0, 3, 1, 2) / 127.5 - 1.0).clamp(-1.0, 1.0)
                t_ids = torch.full((x.shape[0],), task_id, dtype=torch.long, device=device)
                probs = forward_task_embed_probs(model, x, t_ids)
                chunks.append(probs.cpu().numpy())
            all_probs.append(np.concatenate(chunks, axis=0))
    return all_probs


def evaluate_predictions(
    episodes: list[dict[str, Any]],
    probs_list: list[np.ndarray],
    threshold: float | dict[int, float],
) -> dict[str, Any]:
    """Compute whole-episode confusion metrics using the exact ``calibrate_spatial_reward.py`` rule."""
    from calibrate_spatial_reward import confusion, episode_scores

    labels = np.array([bool(ep["success"]) for ep in episodes], dtype=bool)
    init_scores = np.zeros(len(episodes), dtype=np.float32)
    kir_scores = np.zeros(len(episodes), dtype=np.float32)
    final_scores = np.zeros(len(episodes), dtype=np.float32)
    task_ids = np.array([int(ep["task_id"]) for ep in episodes], dtype=np.int64)
    for i, (ep, probs) in enumerate(zip(episodes, probs_list, strict=True)):
        s_init, s_kir, s_final = episode_scores(probs, int(ep["kir_start"]))
        init_scores[i] = s_init
        kir_scores[i] = s_kir
        final_scores[i] = s_final

    def _eval_vec(vec: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
        if isinstance(threshold, dict):
            th = np.array([float(threshold[int(t)]) for t in task_ids[mask]], dtype=np.float32)
            # Shift so comparison >= 0.5 matches vec >= th cleanly.
            shifted = (vec[mask] >= th).astype(np.float32)
            return confusion(labels[mask], shifted, 0.5)
        return confusion(labels[mask], vec[mask], float(threshold))

    all_mask = np.ones(len(episodes), dtype=bool)
    overall = {
        "initial": _eval_vec(init_scores, all_mask),
        "kir": _eval_vec(kir_scores, all_mask),
        "final_frame": _eval_vec(final_scores, all_mask),
    }
    per_task = {}
    for t in range(10):
        t_mask = task_ids == t
        if t_mask.any():
            per_task[str(t)] = _eval_vec(init_scores, t_mask)
    return {
        "overall": overall,
        "per_task": per_task,
        "init_scores": init_scores,
        "kir_scores": kir_scores,
        "final_scores": final_scores,
    }


def select_best_global_threshold(
    episodes: list[dict[str, Any]], probs_list: list[np.ndarray]
) -> float:
    from calibrate_spatial_reward import choose_threshold, episode_scores

    labels = np.array([bool(ep["success"]) for ep in episodes], dtype=bool)
    scores = np.array(
        [episode_scores(probs, int(ep["kir_start"]))[:2] for ep, probs in zip(episodes, probs_list)],
        dtype=np.float32,
    )
    sel = choose_threshold(labels, scores, max_fpr=0.05)
    return float(sel["balanced_accuracy_best"]["threshold"])


def select_per_task_thresholds(
    episodes: list[dict[str, Any]], probs_list: list[np.ndarray], *, fallback: float = 0.9
) -> dict[int, float]:
    from calibrate_spatial_reward import confusion, episode_scores

    thresholds: dict[int, float] = {}
    grid = np.linspace(0.50, 0.98, 49)
    for t in range(10):
        idxs = [i for i, ep in enumerate(episodes) if int(ep["task_id"]) == t]
        labels = np.array([bool(episodes[i]["success"]) for i in idxs], dtype=bool)
        if len(idxs) == 0 or labels.all() or (~labels).all():
            thresholds[t] = float(fallback)
            continue
        scores = np.array(
            [episode_scores(probs_list[i], int(episodes[i]["kir_start"]))[0] for i in idxs],
            dtype=np.float32,
        )
        best = max(
            (
                (
                    confusion(labels, scores, float(th))["balanced_accuracy"],
                    -confusion(labels, scores, float(th))["false_positive_rate"],
                    confusion(labels, scores, float(th))["recall"],
                    -abs(float(th) - fallback),
                    float(th),
                )
                for th in grid
            )
        )
        thresholds[t] = float(best[-1])
    return thresholds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_CALIBRATION_DIR)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--baseline-rm", type=Path, default=DEFAULT_BASELINE_RM)
    parser.add_argument("--model-source", type=Path, default=DEFAULT_MODEL_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--samples-per-task", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--unfreeze-from-block", type=int, default=4)
    parser.add_argument("--threads", type=int, default=48)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    import sys

    if str(EXPERIMENTS) not in sys.path:
        sys.path.insert(0, str(EXPERIMENTS))
    from kir_frame_pool import FramePool

    pool = FramePool(args.pool)
    camera_by_ep = {
        int(row["episode_index"]): row["source_camera_space"] for row in pool.episodes
    }

    episodes_csv = args.calibration_dir / "episodes.csv"
    with episodes_csv.open("r", newline="") as stream:
        all_episodes = list(csv.DictReader(stream))
    for ep in all_episodes:
        ep["episode_index"] = int(ep["episode_index"])
        ep["task_id"] = int(ep["task_id"])
        ep["frames"] = int(ep["frames"])
        ep["kir_start"] = int(ep["kir_start"])
        ep["success"] = ep["success"] == "True"
        ep["source_camera_space"] = camera_by_ep[ep["episode_index"]]

    train_eps = [ep for ep in all_episodes if ep["split"] == "calibration"]
    val_eps = [ep for ep in all_episodes if ep["split"] == "validation"]
    print(
        f"Loaded metadata: {len(train_eps)} train (cal) / {len(val_eps)} val episodes",
        flush=True,
    )

    rm_mod = load_reward_module(args.model_source)

    t0 = time.monotonic()
    frames_by_id: dict[int, np.ndarray] = {}
    libero_to_model_task: dict[int, int] = {}
    libero_task_desc: dict[int, str] = {}
    for idx, ep in enumerate(all_episodes, 1):
        frames, desc = _decode_primary_frames_and_description(
            Path(ep["source"]), ep["source_camera_space"]
        )
        frames_by_id[ep["episode_index"]] = frames
        model_tid = int(rm_mod.instruction_to_task_id(desc, "libero_spatial").item())
        ep["task_description"] = desc
        ep["model_task_id"] = model_tid
        prev = libero_to_model_task.get(ep["task_id"])
        if prev is not None and prev != model_tid:
            raise ValueError(
                f"Inconsistent task_description mapping for libero task {ep['task_id']}: {prev} vs {model_tid}"
            )
        libero_to_model_task[ep["task_id"]] = model_tid
        libero_task_desc[ep["task_id"]] = desc
        if idx % 100 == 0 or idx == len(all_episodes):
            print(
                f"Decoded {idx}/{len(all_episodes)} episodes in {time.monotonic() - t0:.1f}s",
                flush=True,
            )
    print(
        f"Verified libero_task_id -> TaskEmbed model_task_id mapping: {dict(sorted(libero_to_model_task.items()))}",
        flush=True,
    )

    train_frames = [frames_by_id[ep["episode_index"]] for ep in train_eps]
    val_frames = [frames_by_id[ep["episode_index"]] for ep in val_eps]
    all_frames = [frames_by_id[ep["episode_index"]] for ep in all_episodes]

    # Load baseline cached probabilities for zero-leakage comparison & train hard-negative mining.
    probs_dir = args.calibration_dir / "probabilities"
    baseline_val_probs = [
        np.load(probs_dir / f"episode_{ep['episode_index']:04d}.npy", allow_pickle=False)
        for ep in val_eps
    ]
    baseline_all_probs = [
        np.load(probs_dir / f"episode_{ep['episode_index']:04d}.npy", allow_pickle=False)
        for ep in all_episodes
    ]
    base_val_eval = evaluate_predictions(val_eps, baseline_val_probs, 0.9)
    base_all_eval = evaluate_predictions(all_episodes, baseline_all_probs, 0.9)
    print(
        f"Baseline resnet_rm.pth @0.9 | Val(100): bal_acc={base_val_eval['overall']['initial']['balanced_accuracy']:.4f} "
        f"rec={base_val_eval['overall']['initial']['recall']:.4f} fpr={base_val_eval['overall']['initial']['false_positive_rate']:.4f} | "
        f"All(500): bal_acc={base_all_eval['overall']['initial']['balanced_accuracy']:.4f} "
        f"rec={base_all_eval['overall']['initial']['recall']:.4f} fpr={base_all_eval['overall']['initial']['false_positive_rate']:.4f}",
        flush=True,
    )

    pools = build_task_balanced_index(
        train_eps,
        baseline_probs_dir=probs_dir,
        pos_tail_frames=5,
        pre_success_margin=20,
        hard_neg_fp_weight=5.0,
        hard_neg_late_weight=3.0,
    )

    model = rm_mod.TaskEmbedResnetRewModel(
        checkpoint_path=None,
        num_tasks=10,
        task_embed_dim=64,
        task_suite_name="libero_spatial",
    )
    copied = warm_start_task_embed_from_resnet(model, args.baseline_rm)
    set_trainable_layers(model, unfreeze_from_block=args.unfreeze_from_block)
    model = model.to(device)
    print(
        f"Warm-started TaskEmbedResnetRewModel ({copied} backbone tensors); "
        f"unfrozen from block b{args.unfreeze_from_block}",
        flush=True,
    )

    backbone_params = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad and n.startswith("visual_encoder.")
    ]
    head_params = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad and not n.startswith("visual_encoder.")
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.backbone_lr},
            {"params": head_params, "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.backbone_lr * 0.1
    )

    args.output.mkdir(parents=True, exist_ok=True)
    best_score = (-1.0, -1.0, -1.0, -1.0)
    best_state = None
    best_epoch = -1
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        ep_start = time.monotonic()
        records = sample_balanced_epoch(
            pools,
            samples_per_task=args.samples_per_task,
            pos_frac=0.50,
            hard_neg_frac=0.30,
            rng=rng,
        )
        model.train()
        # Keep frozen BatchNorm blocks in eval mode so running stats stay stable.
        for idx, child in enumerate(model.visual_encoder):
            if idx < max(0, args.unfreeze_from_block - 1):
                child.eval()

        loss_sum = 0.0
        n_batches = 0
        for offset in range(0, len(records), args.batch_size):
            chunk = records[offset : offset + args.batch_size]
            batch_np = np.stack([train_frames[e][f] for e, f, _, _ in chunk], axis=0)
            x = torch.from_numpy(batch_np).to(device=device, dtype=torch.float32)
            x = (x.permute(0, 3, 1, 2) / 127.5 - 1.0).clamp(-1.0, 1.0)
            x = _augment_batch_tensor(x)
            t_ids = torch.tensor([t for _, _, t, _ in chunk], dtype=torch.long, device=device)
            y = torch.tensor([lbl for _, _, _, lbl in chunk], dtype=torch.float32, device=device)
            y_smooth = y * (1.0 - args.label_smoothing) + 0.5 * args.label_smoothing

            probs = forward_task_embed_probs(model, x, t_ids).clamp(1e-6, 1.0 - 1e-6)
            loss = F.binary_cross_entropy(probs, y_smooth)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=2.0
            )
            optimizer.step()
            loss_sum += float(loss.item())
            n_batches += 1
        scheduler.step()

        # Whole-episode evaluation on held-out 100 validation episodes.
        val_probs = score_episodes(model, val_eps, val_frames, device=device, batch_size=64)
        val_09 = evaluate_predictions(val_eps, val_probs, 0.9)["overall"]["initial"]
        val_08 = evaluate_predictions(val_eps, val_probs, 0.8)["overall"]["initial"]
        val_05 = evaluate_predictions(val_eps, val_probs, 0.5)["overall"]["initial"]
        # Pick best epoch by held-out val balanced_accuracy @ 0.9 (tie-break lower FPR, higher recall).
        key = (
            round(val_09["balanced_accuracy"], 6),
            -round(val_09["false_positive_rate"], 6),
            round(val_09["recall"], 6),
            round(val_08["balanced_accuracy"], 6),
        )
        if key > best_score:
            best_score = key
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(1, n_batches),
            "val_09_bal_acc": val_09["balanced_accuracy"],
            "val_09_recall": val_09["recall"],
            "val_09_fpr": val_09["false_positive_rate"],
            "val_08_bal_acc": val_08["balanced_accuracy"],
            "val_05_bal_acc": val_05["balanced_accuracy"],
            "seconds": round(time.monotonic() - ep_start, 2),
        }
        history.append(row)
        print(
            f"Epoch {epoch:2d}/{args.epochs} | loss={row['train_loss']:.4f} | "
            f"val@0.9 bal_acc={val_09['balanced_accuracy']:.4f} rec={val_09['recall']:.4f} "
            f"fpr={val_09['false_positive_rate']:.4f} (fp={val_09['fp']}/{val_09['true_failure']}, "
            f"fn={val_09['fn']}/{val_09['true_success']}) | {row['seconds']:.1f}s",
            flush=True,
        )

    assert best_state is not None
    model.load_state_dict(best_state)
    ckpt_path = args.output / "task_balanced_resnet_rm.pth"
    torch.save(best_state, ckpt_path)

    # Calibrate global & per-task thresholds ONLY on the 400 train/calibration episodes.
    train_probs = score_episodes(model, train_eps, train_frames, device=device, batch_size=64)
    val_probs = score_episodes(model, val_eps, val_frames, device=device, batch_size=64)
    all_probs = score_episodes(model, all_episodes, all_frames, device=device, batch_size=64)

    cal_best_th = select_best_global_threshold(train_eps, train_probs)
    cal_per_task_th = select_per_task_thresholds(train_eps, train_probs, fallback=cal_best_th)

    ours_val_09 = evaluate_predictions(val_eps, val_probs, 0.9)
    ours_val_cal = evaluate_predictions(val_eps, val_probs, cal_best_th)
    ours_val_pt = evaluate_predictions(val_eps, val_probs, cal_per_task_th)
    ours_all_09 = evaluate_predictions(all_episodes, all_probs, 0.9)
    ours_all_cal = evaluate_predictions(all_episodes, all_probs, cal_best_th)
    ours_all_pt = evaluate_predictions(all_episodes, all_probs, cal_per_task_th)

    # Write side-by-side per-task comparison CSV (all 500 + val 100).
    csv_path = args.output / "per_task_comparison.csv"
    with csv_path.open("w", newline="") as stream:
        fields = [
            "task_id",
            "model_task_id",
            "task_description",
            "baseline_all_recall",
            "baseline_all_fpr",
            "baseline_all_bal_acc",
            "ours_09_all_recall",
            "ours_09_all_fpr",
            "ours_09_all_bal_acc",
            "ours_cal_all_recall",
            "ours_cal_all_fpr",
            "ours_cal_all_bal_acc",
            "calibrated_task_threshold",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for t in range(10):
            ts = str(t)
            b = base_all_eval["per_task"][ts]
            o9 = ours_all_09["per_task"][ts]
            oc = ours_all_cal["per_task"][ts]
            writer.writerow(
                {
                    "task_id": t,
                    "model_task_id": libero_to_model_task[t],
                    "task_description": libero_task_desc[t],
                    "baseline_all_recall": round(b["recall"], 4),
                    "baseline_all_fpr": round(b["false_positive_rate"], 4),
                    "baseline_all_bal_acc": round(b["balanced_accuracy"], 4),
                    "ours_09_all_recall": round(o9["recall"], 4),
                    "ours_09_all_fpr": round(o9["false_positive_rate"], 4),
                    "ours_09_all_bal_acc": round(o9["balanced_accuracy"], 4),
                    "ours_cal_all_recall": round(oc["recall"], 4),
                    "ours_cal_all_fpr": round(oc["false_positive_rate"], 4),
                    "ours_cal_all_bal_acc": round(oc["balanced_accuracy"], 4),
                    "calibrated_task_threshold": round(cal_per_task_th[t], 4),
                }
            )

    report = {
        "checkpoint": str(ckpt_path),
        "best_epoch": best_epoch,
        "libero_to_model_task_id": {str(k): v for k, v in sorted(libero_to_model_task.items())},
        "calibrated_global_threshold": cal_best_th,
        "calibrated_per_task_thresholds": {str(k): v for k, v in cal_per_task_th.items()},
        "baseline_val_09": base_val_eval["overall"],
        "baseline_val_09_per_task": base_val_eval["per_task"],
        "ours_val_09": ours_val_09["overall"],
        "ours_val_09_per_task": ours_val_09["per_task"],
        "ours_val_cal_global": ours_val_cal["overall"],
        "ours_val_cal_per_task": ours_val_pt["overall"],
        "baseline_all_09": base_all_eval["overall"],
        "ours_all_09": ours_all_09["overall"],
        "ours_all_cal_global": ours_all_cal["overall"],
        "ours_all_cal_per_task": ours_all_pt["overall"],
        "history": history,
    }
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nSaved checkpoint: {ckpt_path}", flush=True)
    print(f"Saved report:     {report_path}", flush=True)
    print(f"Saved per-task:   {csv_path}", flush=True)
    print(
        f"BEST EPOCH {best_epoch} | Calibrated Global Threshold = {cal_best_th:.3f}\n"
        f"  Val(100) Baseline @0.90: bal_acc={base_val_eval['overall']['initial']['balanced_accuracy']:.4f} "
        f"rec={base_val_eval['overall']['initial']['recall']:.4f} fpr={base_val_eval['overall']['initial']['false_positive_rate']:.4f}\n"
        f"  Val(100) Ours     @0.90: bal_acc={ours_val_09['overall']['initial']['balanced_accuracy']:.4f} "
        f"rec={ours_val_09['overall']['initial']['recall']:.4f} fpr={ours_val_09['overall']['initial']['false_positive_rate']:.4f}\n"
        f"  Val(100) Ours   @{cal_best_th:.2f}: bal_acc={ours_val_cal['overall']['initial']['balanced_accuracy']:.4f} "
        f"rec={ours_val_cal['overall']['initial']['recall']:.4f} fpr={ours_val_cal['overall']['initial']['false_positive_rate']:.4f}\n"
        f"  Val(100) Ours  Per-Task: bal_acc={ours_val_pt['overall']['initial']['balanced_accuracy']:.4f} "
        f"rec={ours_val_pt['overall']['initial']['recall']:.4f} fpr={ours_val_pt['overall']['initial']['false_positive_rate']:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
