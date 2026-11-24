"""Unit and integration tests for Tier-2 MERL/HaWMPO chunk hallucination head."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch
import torch.nn as nn

EXPERIMENTS = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENTS.parents[1]
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(REPO_ROOT / "sources/diffsynth"))
sys.path.insert(0, str(REPO_ROOT / "project/third_party/RLinf"))
sys.path.insert(0, str(REPO_ROOT / "sources/rlinf_backend"))

from spatial_wovr_hallucination import (
    ChunkHallucinationTracker,
    ChunkResidualPredictor,
    build_chunk_feature_vector,
    compute_pixel_deltas,
    expected_feature_dim,
    extract_resnet_features_and_probs,
    validate_hallucination_trust_config,
)
from train_chunk_residual_head import (
    compute_chunk_residual_target,
    split_episodes_stratified,
    train_residual_predictor,
)


class _ToyResnetRew(nn.Module):
    """Matches ResnetRewModel's exact 9-stage ``self.net`` layout."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.Identity(),
            nn.Identity(),
            nn.Conv2d(8, 512, 1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )


def test_extract_resnet_features_matches_full_net_probabilities():
    torch.manual_seed(0)
    rm = _ToyResnetRew().eval()
    prev = torch.randn(2, 3, 16, 16).clamp(-1, 1)
    pred = torch.randn(2, 8, 3, 16, 16).clamp(-1, 1)

    with torch.no_grad():
        prev_f, pred_f, probs = extract_resnet_features_and_probs(rm, prev, pred)
        expected = torch.stack([rm.net(pred[i]).squeeze(-1) for i in range(2)], dim=0)

    assert prev_f.shape == (2, 512)
    assert pred_f.shape == (2, 8, 512)
    torch.testing.assert_close(probs, expected, atol=1e-6, rtol=1e-6)


def test_predictor_save_load_and_tracker_merl_hawmpo_behavior(tmp_path):
    torch.manual_seed(1)
    dim = expected_feature_dim(512, 8, 7)
    model = ChunkResidualPredictor(input_dim=dim, hidden_dims=(64, 32), target_scale=0.5)
    ckpt = tmp_path / "head.pt"
    model.save(ckpt, metadata={"val_spearman": 0.85})
    loaded, meta = ChunkResidualPredictor.load(ckpt, device="cpu")
    assert meta["val_spearman"] == pytest.approx(0.85)
    assert float(loaded.target_scale.item()) == pytest.approx(0.5)

    # Constant-output mock predictor to check exact MERL/HaWMPO math.
    class _FixedPredictor(ChunkResidualPredictor):
        def __init__(self, values: torch.Tensor):
            super().__init__(input_dim=dim, hidden_dims=(8,), target_scale=1.0)
            self._values = values

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return self._values.to(features.device)

    # Env 0: low hallucination (eps=0.0); Env 1: high hallucination (eps=3.0).
    fixed = _FixedPredictor(torch.tensor([0.0, 3.0], dtype=torch.float32))
    tracker = ChunkHallucinationTracker(
        fixed, num_envs=2, device="cpu", mode="reward_soft", eta_w=1.0, w_min=0.1
    )
    prev_f = torch.zeros(2, 512)
    pred_f = torch.zeros(2, 8, 512)
    actions = torch.zeros(2, 8, 7)
    probs = torch.full((2, 8), 0.95)
    px = torch.zeros(2, 2)

    trust, hall, eps = tracker.score_chunk(prev_f, pred_f, actions, probs, 8, px)
    assert trust[0].item() == pytest.approx(1.0, abs=1e-6)
    assert trust[1].item() == pytest.approx(0.1, abs=1e-6)  # exp(-3) < 0.1 -> clamped to w_min
    assert hall[0].item() == pytest.approx(0.0, abs=1e-6)
    assert hall[1].item() == pytest.approx(0.9, abs=1e-6)
    assert eps[1].item() == pytest.approx(3.0, abs=1e-6)

    raw_rewards = torch.zeros(2, 8)
    raw_rewards[:, -1] = 5.0
    soft_rewards = tracker.modulate_rewards(raw_rewards, trust)
    assert soft_rewards[0, -1].item() == pytest.approx(5.0, abs=1e-5)
    assert soft_rewards[1, -1].item() == pytest.approx(0.5, abs=1e-5)

    # log_only mode must not alter rewards.
    tracker.mode = "log_only"
    torch.testing.assert_close(tracker.modulate_rewards(raw_rewards, trust), raw_rewards)

    # Reset restores clean per-episode state.
    tracker.reset()
    assert tracker.error_count.sum().item() == 0.0
    assert tracker.last_trust.min().item() == 1.0


def test_train_residual_predictor_learns_and_avoids_episode_leakage(tmp_path):
    rng = np.random.default_rng(7)
    n_eps = 20
    chunks_per_ep = 6
    dim = 64
    features = rng.normal(size=(n_eps * chunks_per_ep, dim)).astype(np.float32)
    true_w = rng.normal(size=(dim,)).astype(np.float32) * 0.25
    logits = features @ true_w + 0.5
    targets = np.log1p(np.exp(logits)).astype(np.float32)  # positive smooth target
    episode_ids = np.repeat(np.arange(n_eps), chunks_per_ep)
    task_ids = np.repeat(np.arange(n_eps) % 5, chunks_per_ep)

    train_idx, val_idx = split_episodes_stratified(episode_ids, task_ids, val_fraction=0.2, seed=42)
    assert set(episode_ids[train_idx]).isdisjoint(set(episode_ids[val_idx]))

    out_ckpt = tmp_path / "chunk_residual_head.pt"
    _, report = train_residual_predictor(
        {
            "features": features,
            "targets": targets,
            "episode_ids": episode_ids,
            "task_ids": task_ids,
        },
        output_path=out_ckpt,
        epochs=60,
        batch_size=32,
        lr=3e-3,
        hidden_dims=(64, 32),
        dropout=0.0,
        device="cpu",
    )
    assert out_ckpt.is_file()
    assert out_ckpt.with_suffix(".report.json").is_file()
    assert report["val_spearman"] > 0.85
    assert report["val_top20_auroc"] > 0.90
    assert report["target_scale_median"] > 0.0


def test_compute_chunk_residual_target_penalizes_false_positives():
    zeros = torch.zeros(8, 3, 8, 8)
    feats = torch.ones(8, 16)
    real_probs = torch.full((8,), 0.1)
    pred_probs_ok = torch.full((8,), 0.12)
    pred_probs_fp = torch.full((8,), 0.95)

    t_ok = compute_chunk_residual_target(
        zeros, zeros, feats, feats, pred_probs_ok, real_probs, episode_success=False
    )
    t_fp = compute_chunk_residual_target(
        zeros, zeros, feats, feats, pred_probs_fp, real_probs, episode_success=False
    )
    assert t_fp > t_ok + 2.0


def test_wan_env_score_classifier_chunk_applies_reward_soft():
    import ast
    import importlib.util

    wm_dir = REPO_ROOT / "project/third_party/RLinf/rlinf/envs/world_model"
    spec = importlib.util.spec_from_file_location(
        "first_success_overlay", wm_dir / "first_success.py"
    )
    first_success_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(first_success_mod)
    FirstSuccessMask = first_success_mod.FirstSuccessMask

    # Load _infer_next_chunk_rewards and _score_classifier_chunk directly from the
    # overlaid WanEnv source, matching how test_spatial_wovr_active_generation tests it.
    wan_source = (wm_dir / "world_model_wan_env.py").read_text()
    tree = ast.parse(wan_source)
    wan_cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "WanEnv"
    )
    ns: dict = {"torch": torch, "np": np, "Optional": object}
    for name in ("_infer_next_chunk_rewards", "_score_classifier_chunk"):
        fn = next(
            n for n in wan_cls.body if isinstance(n, ast.FunctionDef) and n.name == name
        )
        code = compile(ast.Module(body=[fn], type_ignores=[]), "<wan_env>", "exec")
        exec(code, ns)

    class _StubWanEnv:
        _infer_next_chunk_rewards = ns["_infer_next_chunk_rewards"]
        _score_classifier_chunk = ns["_score_classifier_chunk"]

        def _calc_step_reward(self, binary_rewards):
            return binary_rewards * float(self.cfg.reward_coef)

    torch.manual_seed(3)
    env = _StubWanEnv()
    env.num_envs = 2
    env.chunk = 8
    env.device = torch.device("cpu")
    env.elapsed_steps = 16
    env.cfg = OmegaConf.create(
        {
            "reward_model": {"type": "ResnetRewModel", "flip_lr": False},
            "reward_coef": 5.0,
            "use_rel_reward": True,
            "max_episode_steps": 240,
        }
    )
    env.reward_model = _ToyResnetRew().eval()
    # Force high success probability on frame 7 of both envs.
    with torch.no_grad():
        env.reward_model.net[7].weight.zero_()
        env.reward_model.net[7].bias.fill_(5.0)
    env.prev_step_reward = torch.zeros(2, dtype=torch.float32)
    env.first_success_mask = FirstSuccessMask(2, "cpu", threshold=0.9, consecutive_frames=1)
    # current_obs: [B, C, V, T=condition+chunk, H, W]
    env.current_obs = torch.zeros(2, 3, 1, 13, 16, 16, dtype=torch.float32)

    dim = expected_feature_dim(512, 8, 7)

    class _FixedPredictor(ChunkResidualPredictor):
        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return torch.tensor([0.0, 2.0], dtype=torch.float32, device=features.device)

    env.hallucination_tracker = ChunkHallucinationTracker(
        _FixedPredictor(input_dim=dim, hidden_dims=(8,), target_scale=1.0),
        num_envs=2,
        device="cpu",
        mode="reward_soft",
        eta_w=1.0,
        w_min=0.1,
    )
    actions = np.zeros((2, 8, 7), dtype=np.float32)
    probs, rewards, succ, terms, _ = env._score_classifier_chunk(
        all_inactive=False, chunk_actions=actions
    )
    assert succ.all()
    assert terms[:, 0].all()
    # Env 0 (eps=0) keeps full reward 5.0; Env 1 (eps=2) is softened by exp(-2).
    assert rewards[0, 0].item() == pytest.approx(5.0, abs=1e-4)
    assert rewards[1, 0].item() == pytest.approx(5.0 * float(np.exp(-2.0)), abs=1e-4)
    assert rewards[:, 1:].abs().sum().item() == 0.0
    assert env.hallucination_tracker.last_hallucination[1].item() > 0.8


def test_validate_hallucination_trust_config(tmp_path):
    assert validate_hallucination_trust_config(None)["enabled"] is False
    with pytest.raises(ValueError, match="existing checkpoint_path"):
        validate_hallucination_trust_config(
            OmegaConf.create({"enabled": True, "checkpoint_path": str(tmp_path / "missing.pt")})
        )
    ckpt = tmp_path / "exists.pt"
    ckpt.write_bytes(b"ok")
    parsed = validate_hallucination_trust_config(
        OmegaConf.create(
            {
                "enabled": True,
                "checkpoint_path": str(ckpt),
                "mode": "reward_soft",
                "eta_w": 1.5,
                "w_min": 0.15,
                "aggregation": "max",
            }
        )
    )
    assert parsed["enabled"] is True
    assert parsed["eta_w"] == 1.5
    with pytest.raises(ValueError, match="mode"):
        validate_hallucination_trust_config(OmegaConf.create({"mode": "bad"}))
