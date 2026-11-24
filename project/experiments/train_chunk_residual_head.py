#!/usr/bin/env python3
"""Train and calibrate the Tier-2 MERL/HaWMPO chunk hallucination residual head.

Builds chunk-level (feature -> true world-model residual) supervision automatically
from real LIBERO trajectories paired with closed-loop Wan predictions produced by
``wm_eval.py predict --model wan``, reusing ``ResnetRewModel`` visual features.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from spatial_wovr_hallucination import (
    ChunkResidualPredictor,
    build_chunk_feature_vector,
    compute_pixel_deltas,
    extract_resnet_features_and_probs,
)


def _uint8_bhwc_to_norm_bchw(frames: np.ndarray, size: int = 256) -> torch.Tensor:
    """Convert uint8 [N, H, W, 3] RGB to float32 [N, 3, size, size] in [-1, 1]."""
    tensor = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float() / 255.0
    if tuple(tensor.shape[-2:]) != (size, size):
        tensor = F.interpolate(
            tensor, size=(size, size), mode="bilinear", align_corners=False
        )
    return tensor * 2.0 - 1.0


def compute_chunk_residual_target(
    pred_frames: torch.Tensor,
    real_frames: torch.Tensor,
    pred_feats: torch.Tensor,
    real_feats: torch.Tensor,
    pred_probs: torch.Tensor,
    real_probs: torch.Tensor,
    *,
    episode_success: bool,
    lambda_pix: float = 1.0,
    lambda_feat: float = 0.5,
    lambda_rew: float = 2.0,
    lambda_fp: float = 2.0,
    success_threshold: float = 0.9,
) -> float:
    """Compute scalar chunk residual combining visual, feature, and reward error."""
    pix_mse = F.mse_loss(pred_frames, real_frames).item()
    feat_l2 = (
        (pred_feats - real_feats).norm(dim=-1)
        / (real_feats.norm(dim=-1).clamp_min(1e-6))
    ).mean().item()
    rew_err = (pred_probs - real_probs).abs().mean().item()
    pred_pos = bool((pred_probs >= success_threshold).any().item())
    real_pos = bool((real_probs >= success_threshold).any().item())
    false_pos = 1.0 if (pred_pos and (not episode_success or not real_pos)) else 0.0
    return float(
        lambda_pix * pix_mse
        + lambda_feat * feat_l2
        + lambda_rew * rew_err
        + lambda_fp * false_pos
    )


@torch.no_grad()
def build_dataset_from_wm_eval(
    manifest_path: Path,
    wm_eval_dir: Path,
    reward_model: nn.Module,
    *,
    device: str | torch.device = "cpu",
    chunk_size: int = 8,
    max_episode_steps: int = 240,
    lambda_pix: float = 1.0,
    lambda_feat: float = 0.5,
    lambda_rew: float = 2.0,
    lambda_fp: float = 2.0,
    success_threshold: float = 0.9,
) -> dict[str, np.ndarray]:
    """Extract chunk feature/target arrays from ``wm_eval.py`` Wan predictions."""
    from wm_eval_data import read_episode

    manifest = json.loads(manifest_path.read_text())
    reward_model = reward_model.eval().to(device)

    all_features: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_episode_ids: list[int] = []
    all_task_ids: list[int] = []

    used_episodes = 0
    for ep_idx, row in enumerate(manifest["episodes"]):
        pred_npz = wm_eval_dir / "wan" / f"{row['trajectory_id']}.npz"
        if not pred_npz.is_file():
            continue
        primary, _, actions = read_episode(row)
        data = np.load(pred_npz)
        pred_rgb = data["native_prediction"] if "native_prediction" in data else data["prediction"]
        real_rgb = primary[1 : 1 + len(pred_rgb)]
        if len(pred_rgb) < chunk_size or len(real_rgb) != len(pred_rgb):
            continue

        horizon = (len(pred_rgb) // chunk_size) * chunk_size
        init_norm = _uint8_bhwc_to_norm_bchw(primary[:1]).to(device)  # [1, 3, 256, 256]
        pred_norm = _uint8_bhwc_to_norm_bchw(pred_rgb[:horizon]).to(device)
        real_norm = _uint8_bhwc_to_norm_bchw(real_rgb[:horizon]).to(device)

        used_episodes += 1
        for start in range(0, horizon, chunk_size):
            end = start + chunk_size
            prev_frame = init_norm if start == 0 else pred_norm[start - 1 : start]
            pred_chunk = pred_norm[start:end].unsqueeze(0)  # [1, 8, 3, H, W]
            real_chunk = real_norm[start:end].unsqueeze(0)

            prev_feat, pred_feats, pred_probs = extract_resnet_features_and_probs(
                reward_model, prev_frame, pred_chunk
            )
            _, real_feats, real_probs = extract_resnet_features_and_probs(
                reward_model, prev_frame, real_chunk
            )
            px_deltas = compute_pixel_deltas(prev_frame, pred_chunk)
            chunk_actions = actions[start:end][None]  # [1, 8, 7]
            feat_vec = build_chunk_feature_vector(
                prev_feat,
                pred_feats,
                chunk_actions,
                pred_probs,
                elapsed_steps=end,
                pixel_deltas=px_deltas,
                max_episode_steps=max_episode_steps,
                chunk_size=chunk_size,
            )
            target = compute_chunk_residual_target(
                pred_chunk[0],
                real_chunk[0],
                pred_feats[0],
                real_feats[0],
                pred_probs[0],
                real_probs[0],
                episode_success=bool(row.get("success", True)),
                lambda_pix=lambda_pix,
                lambda_feat=lambda_feat,
                lambda_rew=lambda_rew,
                lambda_fp=lambda_fp,
                success_threshold=success_threshold,
            )
            all_features.append(feat_vec[0].cpu().numpy())
            all_targets.append(np.float32(target))
            all_episode_ids.append(ep_idx)
            all_task_ids.append(int(row.get("task_id", 0)))

    if not all_features:
        raise ValueError(
            f"No paired Wan predictions found under {wm_eval_dir / 'wan'} for {manifest_path}"
        )
    return {
        "features": np.stack(all_features, axis=0).astype(np.float32),
        "targets": np.asarray(all_targets, dtype=np.float32),
        "episode_ids": np.asarray(all_episode_ids, dtype=np.int64),
        "task_ids": np.asarray(all_task_ids, dtype=np.int64),
        "used_episodes": np.asarray([used_episodes], dtype=np.int64),
    }


def _spearman_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return 0.0
    r_true = np.argsort(np.argsort(y_true)).astype(np.float64)
    r_pred = np.argsort(np.argsort(y_pred)).astype(np.float64)
    std_t = r_true.std()
    std_p = r_pred.std()
    if std_t < 1e-12 or std_p < 1e-12:
        return 0.0
    return float(np.corrcoef(r_true, r_pred)[0, 1])


def _auroc_top_quantile(
    y_true: np.ndarray, y_pred: np.ndarray, quantile: float = 0.8
) -> float:
    if len(y_true) < 4:
        return 0.5
    thresh = float(np.quantile(y_true, quantile))
    labels = (y_true >= thresh).astype(np.int64)
    pos = labels == 1
    neg = labels == 0
    if not pos.any() or not neg.any():
        return 0.5
    ranks = np.argsort(np.argsort(y_pred)).astype(np.float64) + 1.0
    n_pos = float(pos.sum())
    n_neg = float(neg.sum())
    return float((ranks[pos].sum() - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg))


def split_episodes_stratified(
    episode_ids: np.ndarray,
    task_ids: np.ndarray,
    *,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split chunk indices by episode (never leaking chunks from one episode)."""
    rng = np.random.default_rng(seed)
    ep_to_task: dict[int, int] = {}
    for ep, task in zip(episode_ids.tolist(), task_ids.tolist(), strict=True):
        ep_to_task[ep] = task
    by_task: dict[int, list[int]] = {}
    for ep, task in ep_to_task.items():
        by_task.setdefault(task, []).append(ep)

    val_eps: set[int] = set()
    for task, eps in sorted(by_task.items()):
        ordered = np.asarray(sorted(eps), dtype=np.int64)
        rng.shuffle(ordered)
        n_val = max(1, int(round(len(ordered) * val_fraction))) if len(ordered) > 1 else 0
        val_eps.update(ordered[:n_val].tolist())
    if not val_eps and len(ep_to_task) > 1:
        val_eps.add(sorted(ep_to_task.keys())[-1])

    val_mask = np.isin(episode_ids, list(val_eps))
    train_idx = np.flatnonzero(~val_mask)
    val_idx = np.flatnonzero(val_mask)
    if len(train_idx) == 0:
        raise ValueError("Need at least two episodes to create non-empty train/val split")
    return train_idx, val_idx


def train_residual_predictor(
    dataset: dict[str, np.ndarray],
    *,
    output_path: Path,
    epochs: int = 40,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    hidden_dims: tuple[int, ...] = (256, 64),
    dropout: float = 0.1,
    val_fraction: float = 0.2,
    seed: int = 42,
    device: str | torch.device = "cpu",
) -> tuple[ChunkResidualPredictor, dict[str, Any]]:
    """Train, calibrate, and save a ``ChunkResidualPredictor`` checkpoint."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    features = dataset["features"].astype(np.float32)
    targets = dataset["targets"].astype(np.float32)
    episode_ids = dataset["episode_ids"].astype(np.int64)
    task_ids = dataset["task_ids"].astype(np.int64)

    train_idx, val_idx = split_episodes_stratified(
        episode_ids, task_ids, val_fraction=val_fraction, seed=seed
    )
    train_x = torch.from_numpy(features[train_idx]).to(device)
    train_y = torch.from_numpy(targets[train_idx]).to(device)
    val_x = torch.from_numpy(features[val_idx]).to(device)
    val_y = torch.from_numpy(targets[val_idx]).to(device)

    # Calibrate target_scale to median error so default eta_w=1.0 is well-scaled.
    cal_targets = targets[val_idx] if len(val_idx) > 0 else targets[train_idx]
    target_scale = float(np.median(cal_targets))
    if not math.isfinite(target_scale) or target_scale < 1e-4:
        target_scale = max(float(np.mean(cal_targets)), 1e-3)

    model = ChunkResidualPredictor(
        input_dim=features.shape[1],
        hidden_dims=hidden_dims,
        dropout=dropout,
        target_scale=target_scale,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    best_val_loss = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    history: list[dict[str, float]] = []

    n_train = len(train_x)
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        train_loss_sum = 0.0
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            pred = model(train_x[idx])
            loss = F.smooth_l1_loss(pred, train_y[idx], beta=max(target_scale * 0.5, 1e-3))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss_sum += float(loss.item()) * len(idx)
        scheduler.step()
        train_loss = train_loss_sum / max(1, n_train)

        model.eval()
        with torch.no_grad():
            if len(val_x) > 0:
                val_pred = model(val_x)
                val_loss = float(
                    F.smooth_l1_loss(
                        val_pred, val_y, beta=max(target_scale * 0.5, 1e-3)
                    ).item()
                )
            else:
                val_pred = model(train_x)
                val_loss = train_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch == epochs or epoch % 10 == 0:
            history.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss})

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        eval_x = val_x if len(val_x) > 0 else train_x
        eval_y = (val_y if len(val_x) > 0 else train_y).cpu().numpy()
        eval_pred = model(eval_x).cpu().numpy()

    report = {
        "num_chunks_total": int(len(features)),
        "num_chunks_train": int(len(train_idx)),
        "num_chunks_val": int(len(val_idx)),
        "target_scale_median": target_scale,
        "best_val_loss": best_val_loss,
        "val_spearman": _spearman_corr(eval_y, eval_pred),
        "val_top20_auroc": _auroc_top_quantile(eval_y, eval_pred, quantile=0.8),
        "val_target_mean": float(np.mean(eval_y)),
        "val_pred_mean": float(np.mean(eval_pred)),
        "history": history,
    }
    model.save(output_path, metadata=report)
    report_path = output_path.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return model, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="wm_eval manifest.json path")
    parser.add_argument("--wm-eval-dir", type=Path, help="wm_eval output directory containing wan/*.npz")
    parser.add_argument("--from-dataset-npz", type=Path, help="Optional cached dataset .npz")
    parser.add_argument("--save-dataset-npz", type=Path, help="Optional path to cache extracted dataset .npz")
    parser.add_argument(
        "--resnet-rm",
        type=Path,
        default=SCRIPT_DIR.parents[1] / "assets/wan_baseline/resnet_rm.pth",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output .pt checkpoint path")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.from_dataset_npz:
        raw = np.load(args.from_dataset_npz)
        dataset = {key: raw[key] for key in raw.files}
    else:
        if not args.manifest or not args.wm_eval_dir:
            parser.error("Provide --manifest and --wm-eval-dir (or --from-dataset-npz)")
        diffsynth_root = SCRIPT_DIR.parents[1] / "sources/diffsynth"
        if str(diffsynth_root) not in sys.path:
            sys.path.insert(0, str(diffsynth_root))
        from diffsynth.models.reward_model import ResnetRewModel

        reward_model = ResnetRewModel(checkpoint_path=str(args.resnet_rm))
        dataset = build_dataset_from_wm_eval(
            args.manifest,
            args.wm_eval_dir,
            reward_model,
            device=args.device,
        )
        if args.save_dataset_npz:
            args.save_dataset_npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.save_dataset_npz, **dataset)

    _, report = train_residual_predictor(
        dataset,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_fraction=args.val_fraction,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
