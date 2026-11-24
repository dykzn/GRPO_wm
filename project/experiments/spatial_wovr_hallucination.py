"""MERL / HaWMPO chunk-level hallucination & residual trust head for WanEnv."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

RESIDUAL_SCHEMA = "spatial-wovr-chunk-residual-v1"
SUPPORTED_TRUST_MODES = frozenset({"reward_soft", "log_only"})
SUPPORTED_AGGREGATIONS = frozenset({"current", "mean", "max", "blend"})


def expected_feature_dim(
    visual_dim: int = 512, chunk_size: int = 8, action_dim: int = 7
) -> int:
    """Return input dimension of ``ChunkResidualPredictor``."""
    # [prev_feat, pred_mean, pred_last, pred_last - prev_feat]
    # + [flat actions (chunk*action_dim), action_kinematics (4)]
    # + [reward_summary (4), rollout_depth (2), pixel_delta (2)]
    return visual_dim * 4 + chunk_size * action_dim + 4 + 4 + 2 + 2


def validate_hallucination_trust_config(cfg: Any) -> dict[str, Any]:
    """Validate ``env.train.hallucination_trust`` configuration."""
    if cfg is None:
        return {
            "enabled": False,
            "checkpoint_path": None,
            "mode": "reward_soft",
            "eta_w": 1.0,
            "w_min": 0.1,
            "aggregation": "blend",
            "blend_current": 0.5,
        }
    enabled = bool(cfg.get("enabled", False))
    checkpoint_path = cfg.get("checkpoint_path", None)
    mode = str(cfg.get("mode", "reward_soft")).lower()
    aggregation = str(cfg.get("aggregation", "blend")).lower()
    eta_w = float(cfg.get("eta_w", 1.0))
    w_min = float(cfg.get("w_min", 0.1))
    blend_current = float(cfg.get("blend_current", 0.5))

    if mode not in SUPPORTED_TRUST_MODES:
        raise ValueError(
            f"hallucination_trust.mode must be in {sorted(SUPPORTED_TRUST_MODES)}, got {mode!r}"
        )
    if aggregation not in SUPPORTED_AGGREGATIONS:
        raise ValueError(
            "hallucination_trust.aggregation must be in "
            f"{sorted(SUPPORTED_AGGREGATIONS)}, got {aggregation!r}"
        )
    if not math.isfinite(eta_w) or eta_w < 0.0:
        raise ValueError("hallucination_trust.eta_w must be finite and >= 0")
    if not math.isfinite(w_min) or not (0.0 < w_min <= 1.0):
        raise ValueError("hallucination_trust.w_min must be in (0, 1]")
    if not math.isfinite(blend_current) or not (0.0 <= blend_current <= 1.0):
        raise ValueError("hallucination_trust.blend_current must be in [0, 1]")

    if enabled:
        if not checkpoint_path or not Path(str(checkpoint_path)).is_file():
            raise ValueError(
                "hallucination_trust.enabled=true requires an existing "
                f"checkpoint_path, got {checkpoint_path!r}"
            )
    return {
        "enabled": enabled,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "mode": mode,
        "eta_w": eta_w,
        "w_min": w_min,
        "aggregation": aggregation,
        "blend_current": blend_current,
    }


def extract_resnet_features_and_probs(
    reward_model: nn.Module,
    prev_frames: torch.Tensor,
    pred_frames: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract penultimate ResNet features and reward probabilities in one pass.

    Args:
        reward_model: ``ResnetRewModel`` whose ``net`` ends with
            ``AdaptiveAvgPool2d -> Flatten -> Linear(512, 1) -> Sigmoid``.
        prev_frames: ``[B, 3, H, W]`` in ``[-1, 1]`` (last frame before chunk).
        pred_frames: ``[B, T, 3, H, W]`` in ``[-1, 1]`` (predicted chunk frames).

    Returns:
        ``(prev_feat [B, 512], pred_feats [B, T, 512], probs [B, T])``.
    """
    net = reward_model.net
    backbone = net[:7]
    head = net[7:]
    bsz, chunk_len = pred_frames.shape[:2]

    prev_feats = []
    pred_feats_list = []
    probs_list = []
    for env_idx in range(bsz):
        # Process one environment's 1 + T frames at a time to bound activation VRAM.
        frames = torch.cat(
            [prev_frames[env_idx : env_idx + 1], pred_frames[env_idx]], dim=0
        )
        frames = frames.clamp(-1.0, 1.0).to(dtype=torch.float32)
        feats = backbone(frames)  # [1 + T, 512]
        probs = head(feats[1:]).squeeze(-1)  # [T]
        prev_feats.append(feats[0])
        pred_feats_list.append(feats[1:])
        probs_list.append(probs)

    return (
        torch.stack(prev_feats, dim=0),
        torch.stack(pred_feats_list, dim=0),
        torch.stack(probs_list, dim=0),
    )


def compute_pixel_deltas(
    prev_frames: torch.Tensor, pred_frames: torch.Tensor
) -> torch.Tensor:
    """Compute [mean_abs_diff_from_prev, mean_abs_consecutive_diff] per env."""
    prev = prev_frames.clamp(-1.0, 1.0).to(dtype=torch.float32)
    pred = pred_frames.clamp(-1.0, 1.0).to(dtype=torch.float32)
    diff_from_prev = (pred - prev[:, None]).abs().mean(dim=(1, 2, 3, 4))
    if pred.shape[1] > 1:
        consec = (pred[:, 1:] - pred[:, :-1]).abs().mean(dim=(1, 2, 3, 4))
    else:
        consec = diff_from_prev
    return torch.stack([diff_from_prev, consec], dim=-1)


def build_chunk_feature_vector(
    prev_feat: torch.Tensor,
    pred_feats: torch.Tensor,
    actions: torch.Tensor | np.ndarray,
    probs: torch.Tensor,
    elapsed_steps: torch.Tensor | int | float,
    pixel_deltas: torch.Tensor,
    *,
    max_episode_steps: int = 240,
    chunk_size: int = 8,
    action_dim: int = 7,
) -> torch.Tensor:
    """Assemble the fixed-width chunk feature vector for residual prediction."""
    device = prev_feat.device
    bsz = prev_feat.shape[0]
    prev_f = prev_feat.to(device=device, dtype=torch.float32)
    pred_f = pred_feats.to(device=device, dtype=torch.float32)
    pred_mean = pred_f.mean(dim=1)
    pred_last = pred_f[:, -1]
    feat_delta = pred_last - prev_f

    if isinstance(actions, np.ndarray):
        act = torch.from_numpy(actions).to(device=device, dtype=torch.float32)
    else:
        act = actions.to(device=device, dtype=torch.float32)
    if act.shape != (bsz, chunk_size, action_dim):
        raise ValueError(
            f"Expected actions shape {(bsz, chunk_size, action_dim)}, got {tuple(act.shape)}"
        )
    act_flat = act.reshape(bsz, chunk_size * action_dim)
    xyz_norm = act[..., :3].norm(dim=-1)
    mean_xyz = xyz_norm.mean(dim=-1)
    max_xyz = xyz_norm.amax(dim=-1)
    grip_delta = (act[:, 1:, -1] - act[:, :-1, -1]).abs().mean(dim=-1) if chunk_size > 1 else torch.zeros(bsz, device=device)
    total_norm = act.norm(dim=-1).mean(dim=-1)
    act_kin = torch.stack([mean_xyz, max_xyz, grip_delta, total_norm], dim=-1)

    probs_f = probs.to(device=device, dtype=torch.float32)
    p_mean = probs_f.mean(dim=-1)
    p_max = probs_f.amax(dim=-1)
    p_std = probs_f.std(dim=-1, unbiased=False)
    p_trend = probs_f[:, -1] - probs_f[:, 0]
    rew_summary = torch.stack([p_mean, p_max, p_std, p_trend], dim=-1)

    if isinstance(elapsed_steps, torch.Tensor):
        steps = elapsed_steps.to(device=device, dtype=torch.float32).reshape(bsz)
    else:
        steps = torch.full((bsz,), float(elapsed_steps), device=device, dtype=torch.float32)
    max_steps = max(1.0, float(max_episode_steps))
    depth = torch.stack(
        [steps / max_steps, (steps / float(chunk_size)) / (max_steps / float(chunk_size))],
        dim=-1,
    )

    px = pixel_deltas.to(device=device, dtype=torch.float32)
    return torch.cat(
        [prev_f, pred_mean, pred_last, feat_delta, act_flat, act_kin, rew_summary, depth, px],
        dim=-1,
    )


class ChunkResidualPredictor(nn.Module):
    """Lightweight MLP estimating per-chunk world-model residual error eps_hat >= 0."""

    def __init__(
        self,
        input_dim: int | None = None,
        hidden_dims: tuple[int, ...] = (256, 64),
        dropout: float = 0.1,
        target_scale: float = 1.0,
        visual_dim: int = 512,
        chunk_size: int = 8,
        action_dim: int = 7,
    ) -> None:
        super().__init__()
        self.visual_dim = int(visual_dim)
        self.chunk_size = int(chunk_size)
        self.action_dim = int(action_dim)
        self.input_dim = (
            int(input_dim)
            if input_dim is not None
            else expected_feature_dim(self.visual_dim, self.chunk_size, self.action_dim)
        )
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.dropout = float(dropout)
        self.register_buffer(
            "target_scale", torch.tensor(float(max(target_scale, 1e-6)), dtype=torch.float32)
        )

        layers: list[nn.Module] = [nn.LayerNorm(self.input_dim)]
        in_d = self.input_dim
        for h in self.hidden_dims:
            layers.append(nn.Linear(in_d, h))
            layers.append(nn.SiLU())
            if self.dropout > 0.0:
                layers.append(nn.Dropout(self.dropout))
            in_d = h
        layers.append(nn.Linear(in_d, 1))
        layers.append(nn.Softplus())
        self.mlp = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.mlp(features).squeeze(-1)

    def config_dict(self) -> dict[str, Any]:
        return {
            "schema": RESIDUAL_SCHEMA,
            "input_dim": self.input_dim,
            "hidden_dims": list(self.hidden_dims),
            "dropout": self.dropout,
            "target_scale": float(self.target_scale.item()),
            "visual_dim": self.visual_dim,
            "chunk_size": self.chunk_size,
            "action_dim": self.action_dim,
        }

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": RESIDUAL_SCHEMA,
            "config": self.config_dict(),
            "state_dict": {k: v.detach().cpu() for k, v in self.state_dict().items()},
            "metadata": metadata or {},
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)

    @classmethod
    def load(
        cls, path: str | Path, device: str | torch.device = "cpu"
    ) -> tuple["ChunkResidualPredictor", dict[str, Any]]:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if payload.get("schema") != RESIDUAL_SCHEMA:
            raise ValueError(
                f"Unsupported residual predictor schema: {payload.get('schema')!r}"
            )
        cfg = payload["config"]
        model = cls(
            input_dim=cfg["input_dim"],
            hidden_dims=tuple(cfg["hidden_dims"]),
            dropout=cfg.get("dropout", 0.1),
            target_scale=cfg.get("target_scale", 1.0),
            visual_dim=cfg.get("visual_dim", 512),
            chunk_size=cfg.get("chunk_size", 8),
            action_dim=cfg.get("action_dim", 7),
        )
        model.load_state_dict(payload["state_dict"])
        model.eval().to(device)
        return model, payload.get("metadata", {})


class ChunkHallucinationTracker:
    """Stateful per-environment MERL/HaWMPO chunk trust evaluator for ``WanEnv``."""

    def __init__(
        self,
        predictor: ChunkResidualPredictor,
        num_envs: int,
        device: str | torch.device,
        *,
        mode: str = "reward_soft",
        eta_w: float = 1.0,
        w_min: float = 0.1,
        aggregation: str = "blend",
        blend_current: float = 0.5,
        max_episode_steps: int = 240,
        chunk_size: int = 8,
        action_dim: int = 7,
    ) -> None:
        self.predictor = predictor.eval().to(device)
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.mode = mode
        self.eta_w = float(eta_w)
        self.w_min = float(w_min)
        self.aggregation = aggregation
        self.blend_current = float(blend_current)
        self.max_episode_steps = int(max_episode_steps)
        self.chunk_size = int(chunk_size)
        self.action_dim = int(action_dim)

        self.error_sum = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.error_count = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.error_max = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.last_residual = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.last_trust = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
        self.last_hallucination = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )

    def reset(self, env_idx: Any = None) -> None:
        idx = slice(None) if env_idx is None else env_idx
        self.error_sum[idx] = 0.0
        self.error_count[idx] = 0.0
        self.error_max[idx] = 0.0
        self.last_residual[idx] = 0.0
        self.last_trust[idx] = 1.0
        self.last_hallucination[idx] = 0.0

    def to(self, device: str | torch.device) -> None:
        self.device = torch.device(device)
        self.predictor = self.predictor.to(self.device)
        self.error_sum = self.error_sum.to(self.device)
        self.error_count = self.error_count.to(self.device)
        self.error_max = self.error_max.to(self.device)
        self.last_residual = self.last_residual.to(self.device)
        self.last_trust = self.last_trust.to(self.device)
        self.last_hallucination = self.last_hallucination.to(self.device)

    @torch.no_grad()
    def score_chunk(
        self,
        prev_feat: torch.Tensor,
        pred_feats: torch.Tensor,
        actions: torch.Tensor | np.ndarray,
        probs: torch.Tensor,
        elapsed_steps: torch.Tensor | int | float,
        pixel_deltas: torch.Tensor,
        active_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict chunk residual error and compute MERL trust / HaWMPO hallucination."""
        features = build_chunk_feature_vector(
            prev_feat,
            pred_feats,
            actions,
            probs,
            elapsed_steps,
            pixel_deltas,
            max_episode_steps=self.max_episode_steps,
            chunk_size=self.chunk_size,
            action_dim=self.action_dim,
        ).to(self.device)
        raw_eps = self.predictor(features)
        scale = float(self.predictor.target_scale.item())
        norm_eps = raw_eps / max(scale, 1e-6)

        if active_mask is None:
            active = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            active = active_mask.to(device=self.device, dtype=torch.bool)

        new_sum = torch.where(active, self.error_sum + norm_eps, self.error_sum)
        new_count = torch.where(active, self.error_count + 1.0, self.error_count)
        new_max = torch.where(active, torch.maximum(self.error_max, norm_eps), self.error_max)
        self.error_sum.copy_(new_sum)
        self.error_count.copy_(new_count)
        self.error_max.copy_(new_max)

        prefix_mean = self.error_sum / self.error_count.clamp_min(1.0)
        if self.aggregation == "current":
            eff_eps = norm_eps
        elif self.aggregation == "mean":
            eff_eps = prefix_mean
        elif self.aggregation == "max":
            eff_eps = self.error_max
        else:  # blend
            eff_eps = (
                self.blend_current * norm_eps + (1.0 - self.blend_current) * prefix_mean
            )

        trust = torch.exp(-self.eta_w * eff_eps).clamp(min=self.w_min, max=1.0)
        trust = torch.where(active, trust, self.last_trust)
        hallucination = 1.0 - trust

        self.last_residual.copy_(torch.where(active, raw_eps, self.last_residual))
        self.last_trust.copy_(trust)
        self.last_hallucination.copy_(hallucination)
        return trust, hallucination, self.last_residual

    def modulate_rewards(
        self, rewards: torch.Tensor, trust: torch.Tensor
    ) -> torch.Tensor:
        """Apply Reward-Soft modulation when ``mode == 'reward_soft'``."""
        if self.mode != "reward_soft":
            return rewards
        return rewards * trust.to(device=rewards.device, dtype=rewards.dtype)[:, None]
