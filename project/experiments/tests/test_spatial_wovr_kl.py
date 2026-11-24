"""Unit and integration tests for embodied reference-policy KL regularization."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import sys

from omegaconf import OmegaConf
import pytest
import torch
import torch.nn as nn

EXPERIMENTS = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENTS.parents[1]
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(REPO_ROOT / "sources/rlinf_backend"))

from spatial_wovr_kl import (
    compute_embodied_kl_loss,
    compute_ref_logprobs,
    snapshot_trainable_weights,
    swap_trainable_weights,
    validate_kl_config,
)
import spatial_wovr_pair_launch as pair_launch


class _ToyPolicy(nn.Module):
    """Minimal policy mimicking Pi0.5 train_expert_only=True (frozen VLM + trainable expert)."""

    def __init__(self, in_dim: int = 4, action_chunk: int = 8, action_dim: int = 7):
        super().__init__()
        self.frozen_vlm = nn.Linear(in_dim, in_dim, bias=False)
        for p in self.frozen_vlm.parameters():
            p.requires_grad = False
        self.expert = nn.Linear(in_dim, action_chunk * action_dim)
        self.action_chunk = action_chunk
        self.action_dim = action_dim

    def forward(self, forward_inputs, **kwargs):
        x = self.frozen_vlm(forward_inputs["obs"])
        out = self.expert(x).reshape(-1, self.action_chunk, self.action_dim)
        # Negative squared error mimics Gaussian flow_sde log-probability.
        logprobs = -0.5 * (forward_inputs["target"] - out).square()
        return {"logprobs": logprobs}


def test_validate_kl_config_rules():
    assert validate_kl_config(OmegaConf.create({"kl_beta": 0.0, "kl_penalty": "kl"})) == (
        0.0,
        "kl",
    )
    assert validate_kl_config(
        OmegaConf.create({"kl_beta": 0.01, "kl_penalty": "low_var_kl"})
    ) == (0.01, "low_var_kl")
    assert validate_kl_config(
        OmegaConf.create({"kl_beta": 0.02, "kl_penalty": "k3"})
    ) == (0.02, "k3")
    assert validate_kl_config(
        OmegaConf.create({"kl_beta": 0.05, "kl_penalty": "mse"})
    ) == (0.05, "mse")

    with pytest.raises(ValueError, match="Direct actor KL loss requires"):
        validate_kl_config(OmegaConf.create({"kl_beta": 0.01, "kl_penalty": "kl"}))
    with pytest.raises(ValueError, match="Direct actor KL loss requires"):
        validate_kl_config(OmegaConf.create({"kl_beta": 0.01, "kl_penalty": "k1"}))
    with pytest.raises(ValueError, match="nonnegative"):
        validate_kl_config(OmegaConf.create({"kl_beta": -0.01, "kl_penalty": "low_var_kl"}))
    with pytest.raises(ValueError, match="nonnegative"):
        validate_kl_config(
            OmegaConf.create({"kl_beta": float("nan"), "kl_penalty": "low_var_kl"})
        )


def test_snapshot_and_swap_preserves_parameter_identity_and_restores_on_error():
    torch.manual_seed(0)
    model = _ToyPolicy()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    ref_weights = snapshot_trainable_weights(model)

    # Frozen parameters must not be snapshotted.
    assert set(ref_weights.keys()) == {"expert.weight", "expert.bias"}
    orig_param_ids = {id(p) for p in model.parameters() if p.requires_grad}

    # Modify trainable weights via an optimizer step.
    ref_w_before = model.expert.weight.detach().clone()
    x = torch.randn(2, 4)
    loss = model({"obs": x, "target": torch.zeros(2, 8, 7)})["logprobs"].mean()
    (-loss).backward()
    opt.step()
    opt.zero_grad()
    updated_w = model.expert.weight.detach().clone()
    assert not torch.allclose(updated_w, ref_w_before)

    with swap_trainable_weights(model, ref_weights):
        torch.testing.assert_close(model.expert.weight.detach(), ref_w_before)
        assert {id(p) for p in model.parameters() if p.requires_grad} == orig_param_ids
    torch.testing.assert_close(model.expert.weight.detach(), updated_w)

    # Restore even if body raises.
    with pytest.raises(RuntimeError, match="boom"):
        with swap_trainable_weights(model, ref_weights):
            torch.testing.assert_close(model.expert.weight.detach(), ref_w_before)
            raise RuntimeError("boom")
    torch.testing.assert_close(model.expert.weight.detach(), updated_w)
    assert {id(p) for p in model.parameters() if p.requires_grad} == orig_param_ids


def test_compute_embodied_kl_loss_properties_and_gradients():
    ref = torch.tensor([[[-1.0, -0.5], [-0.2, -0.8]]], dtype=torch.float32)
    mask = torch.tensor([[True, False]])
    mask_sum = torch.tensor([[10.0, 10.0]])

    # Identical logprobs -> zero KL and zero gradient.
    lp_same = ref.clone().requires_grad_(True)
    loss_same, metrics_same = compute_embodied_kl_loss(
        lp_same,
        ref,
        kl_beta=0.01,
        kl_penalty_type="low_var_kl",
        logprob_type="token_level",
        reward_type="action_level",
        single_action_dim=2,
        loss_mask=mask,
        loss_mask_sum=mask_sum,
        max_episode_steps=20,
    )
    assert loss_same.item() == pytest.approx(0.0, abs=1e-7)
    assert metrics_same["actor/kl_loss"] == pytest.approx(0.0, abs=1e-7)
    loss_same.backward()
    torch.testing.assert_close(lp_same.grad, torch.zeros_like(lp_same))

    # Positive & negative deviations both yield positive KL and pull toward ref.
    lp_diff = (ref + torch.tensor([[[0.4, -0.3], [5.0, -5.0]]])).requires_grad_(True)
    loss_diff, metrics_diff = compute_embodied_kl_loss(
        lp_diff,
        ref,
        kl_beta=0.01,
        kl_penalty_type="low_var_kl",
        logprob_type="token_level",
        reward_type="action_level",
        single_action_dim=2,
        loss_mask=mask,
        loss_mask_sum=mask_sum,
        max_episode_steps=20,
    )
    assert loss_diff.item() > 0.0
    assert metrics_diff["actor/kl_penalty_loss"] == pytest.approx(
        0.01 * metrics_diff["actor/kl_loss"], rel=1e-6
    )
    loss_diff.backward()
    # Active action 0: +0.4 deviation -> positive grad (descent decreases lp);
    # -0.3 deviation -> negative grad (descent increases lp).
    assert lp_diff.grad[0, 0, 0].item() > 0.0
    assert lp_diff.grad[0, 0, 1].item() < 0.0
    # Masked-out action 1 must receive exact zero gradient.
    torch.testing.assert_close(lp_diff.grad[0, 1], torch.zeros(2))

    # Empty mask -> zero loss, finite zero grad.
    lp_empty = (ref + 0.5).requires_grad_(True)
    loss_empty, metrics_empty = compute_embodied_kl_loss(
        lp_empty,
        ref,
        kl_beta=0.01,
        kl_penalty_type="low_var_kl",
        logprob_type="token_level",
        reward_type="action_level",
        single_action_dim=2,
        loss_mask=torch.zeros_like(mask),
        loss_mask_sum=mask_sum,
        max_episode_steps=20,
    )
    assert loss_empty.item() == 0.0
    assert metrics_empty["actor/ref_kl_mean"] == 0.0
    loss_empty.backward()
    torch.testing.assert_close(lp_empty.grad, torch.zeros_like(lp_empty))


def test_compute_ref_logprobs_uses_reference_weights_and_restores_current():
    torch.manual_seed(1)
    model = _ToyPolicy()
    ref_weights = snapshot_trainable_weights(model)
    with torch.no_grad():
        model.expert.weight.add_(0.25)

    T, B = 2, 4
    obs = torch.randn(T, B, 4)
    target = torch.randn(T, B, 8, 7)
    rollout_batch = {
        "prev_logprobs": torch.zeros(T, B, 8, 7),
        "forward_inputs": {"obs": obs, "target": target},
    }
    model.train()
    ref_lp = compute_ref_logprobs(
        model,
        rollout_batch,
        ref_weights,
        device="cpu",
        micro_batch_size=3,
        amp_context=nullcontext(),
    )
    assert model.training is True
    assert ref_lp.shape == (T, B, 8, 7)

    # Compare against explicit ref model forward.
    ref_model = _ToyPolicy()
    with torch.no_grad():
        ref_model.frozen_vlm.weight.copy_(model.frozen_vlm.weight)
        ref_model.expert.weight.copy_(ref_weights["expert.weight"])
        ref_model.expert.bias.copy_(ref_weights["expert.bias"])
        expected = ref_model(
            {"obs": obs.reshape(-1, 4), "target": target.reshape(-1, 8, 7)}
        )["logprobs"].reshape(T, B, 8, 7)
    torch.testing.assert_close(ref_lp, expected)

    # Current model weights remain shifted (not overwritten by ref).
    with torch.no_grad():
        current_lp = model(
            {"obs": obs.reshape(-1, 4), "target": target.reshape(-1, 8, 7)}
        )["logprobs"].reshape(T, B, 8, 7)
    assert not torch.allclose(current_lp, ref_lp)


def test_actor_load_checkpoint_anchors_kl_and_backprops_kl_gradient():
    from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
    from spatial_wovr_actor import WorldModelFSDPActor

    torch.manual_seed(2)
    actor = object.__new__(WorldModelFSDPActor)
    actor.cfg = OmegaConf.create(
        {
            "runner": {"task_type": "embodied", "resume_dir": "/tmp/fake_step_60"},
            "env": {"train": {"max_episode_steps": 240}},
            "actor": {
                "micro_batch_size": 2,
                "model": {"model_type": "openpi", "action_dim": 7},
                "optim": {"lr": 5.0e-6, "use_config_lr_on_resume": False},
            },
            "algorithm": {
                "adv_type": "grpo",
                "loss_type": "actor",
                "logprob_type": "token_level",
                "reward_type": "action_level",
                "entropy_type": "token_level",
                "entropy_bonus": 0.0,
                "clip_ratio_high": 0.28,
                "clip_ratio_low": 0.2,
                "kl_penalty": "low_var_kl",
                "kl_beta": 0.05,
            },
        }
    )
    actor.device = "cpu"
    actor.amp_context = nullcontext()
    actor.model = _ToyPolicy()
    actor.optimizer_steps = 0
    actor.critic_warmup_steps = 0
    actor.enable_sft_co_train = False
    actor.gradient_accumulation = 1
    actor.grad_scaler = torch.amp.GradScaler(enabled=False)
    actor.before_micro_batch = lambda model, is_last_micro_batch: nullcontext()
    actor._kl_beta, actor._kl_penalty_type = validate_kl_config(actor.cfg.algorithm)
    actor._ref_trainable_weights = None
    actor._active_gpu_peak_profile = None
    actor._active_training_diagnostics = None
    actor._shared_wan_owner = None

    # Simulate resuming from step 60 by snapping ref weights after checkpoint load.
    actor._ref_trainable_weights = snapshot_trainable_weights(actor.model)
    with torch.no_grad():
        actor.model.expert.weight.add_(0.2)

    obs = torch.randn(2, 4)
    target = torch.randn(2, 8, 7)
    with torch.no_grad():
        prev_lp = actor.model({"obs": obs, "target": target})["logprobs"].detach()
    ref_lp = compute_ref_logprobs(
        actor.model,
        {
            "prev_logprobs": prev_lp.unsqueeze(0),
            "forward_inputs": {"obs": obs.unsqueeze(0), "target": target.unsqueeze(0)},
        },
        actor._ref_trainable_weights,
        device="cpu",
        micro_batch_size=2,
        amp_context=nullcontext(),
    ).squeeze(0)

    # Zero advantages isolate the pure KL gradient contribution.
    micro_batch = {
        "advantages": torch.zeros(2, 8),
        "prev_logprobs": prev_lp,
        "ref_logprobs": ref_lp,
        "loss_mask": torch.ones(2, 8, dtype=torch.bool),
        "loss_mask_sum": torch.full((2, 8), 8.0),
        "forward_inputs": {"obs": obs, "target": target},
    }
    metrics: dict[str, list[float]] = {}
    actor.train_micro_batch(micro_batch, metrics, is_last=True)

    assert metrics["actor/kl_loss"][0] > 0.0
    assert metrics["actor/kl_penalty_loss"][0] == pytest.approx(
        0.05 * metrics["actor/kl_loss"][0], rel=1e-6
    )
    assert metrics["actor/ref_kl_mean"][0] > 0.0
    assert actor.model.expert.weight.grad is not None
    assert actor.model.expert.weight.grad.abs().sum().item() > 0.0
    # Sanity check: native class still exists untouched.
    assert issubclass(WorldModelFSDPActor, EmbodiedFSDPActor)


def test_compose_saved_config_reverts_lr_and_enables_kl(monkeypatch, tmp_path):
    monkeypatch.setenv("SPATIAL_WOVR_OUTPUT", str(tmp_path / "out"))
    monkeypatch.setenv("WAN_SPATIAL_CHECKPOINT", str(tmp_path / "wan"))
    monkeypatch.setenv("SPATIAL_WOVR_GPUS", "0,1,2,3,4,5,6,7")
    monkeypatch.setenv("SPATIAL_COSMOS_PORT", "18759")

    template = REPO_ROOT / "config/spatial.reference.yaml"

    # Without OPENVLA profile
    monkeypatch.delenv("SPATIAL_WOVR_OPENVLA_WAN_ACTOR", raising=False)
    cfg_default = pair_launch.compose_saved_config(template, [])
    assert float(cfg_default.actor.optim.lr) == pytest.approx(5.0e-6)
    assert cfg_default.algorithm.kl_penalty == "low_var_kl"
    assert float(cfg_default.algorithm.kl_beta) == pytest.approx(0.01)

    # With OPENVLA_WAN_ACTOR profile (the exact mode used by grpo60_to80_kir50_grace16_r3)
    monkeypatch.setenv("SPATIAL_WOVR_OPENVLA_WAN_ACTOR", "1")
    cfg_profile = pair_launch.compose_saved_config(template, [])
    assert float(cfg_profile.actor.optim.lr) == pytest.approx(5.0e-6)
    assert cfg_profile.actor.optim.use_config_lr_on_resume is True
    assert cfg_profile.algorithm.kl_penalty == "low_var_kl"
    assert float(cfg_profile.algorithm.kl_beta) == pytest.approx(0.01)

    # CLI override still works (e.g., turning KL off or tuning beta).
    cfg_override = pair_launch.compose_saved_config(
        template, ["algorithm.kl_beta=0.0", "actor.global_batch_size=2048"]
    )
    assert float(cfg_override.algorithm.kl_beta) == 0.0
    assert int(cfg_override.actor.global_batch_size) == 2048
