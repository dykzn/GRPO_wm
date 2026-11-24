"""Unit and integration tests for WorldSample Route-A Hybrid PPO + GRPO."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest
import torch
import torch.nn as nn

EXPERIMENTS = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENTS.parents[1]
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(REPO_ROOT / "sources/rlinf_backend"))

import rlinf.algorithms  # noqa: F401  # register policy losses & advantages
from spatial_wovr_actor import WorldModelFSDPActor
from spatial_wovr_hybrid import (
    HybridPpoGrpoConfig,
    HybridRunnerMixin,
    compute_paired_step_schedule,
    compute_wm_rho,
    install_value_head_checkpoint_compat,
    load_state_dict_allow_missing_value_head,
    merge_real_and_wm_gradients,
    parse_hybrid_config,
    snapshot_param_grads_to_cpu,
    validate_hybrid_config,
)
from spatial_wovr_kl import snapshot_trainable_weights, swap_trainable_weights


class _ToyActorCriticPolicy(nn.Module):
    """Minimal Pi0.5-like actor+critic with frozen VLM, trainable expert, detached value head."""

    def __init__(self, in_dim: int = 4, action_chunk: int = 8, action_dim: int = 7):
        super().__init__()
        self.frozen_vlm = nn.Linear(in_dim, in_dim, bias=False)
        for p in self.frozen_vlm.parameters():
            p.requires_grad = False
        self.expert = nn.Linear(in_dim, action_chunk * action_dim)
        self.value_head = nn.Linear(in_dim, 1)
        self.action_chunk = action_chunk
        self.action_dim = action_dim

    def forward(
        self,
        forward_inputs,
        *,
        compute_logprobs: bool = True,
        compute_entropy: bool = False,
        compute_values: bool = False,
        **kwargs,
    ):
        x = self.frozen_vlm(forward_inputs["obs"])
        pred = self.expert(x).reshape(-1, self.action_chunk, self.action_dim)
        out = {}
        if compute_logprobs:
            out["logprobs"] = -0.5 * (forward_inputs["target"] - pred).square()
        if compute_entropy:
            # Positive per-token proxy entropy depending smoothly on prediction spread.
            out["entropy"] = 0.5 + 0.1 * pred.detach().abs()
        if compute_values:
            out["values"] = self.value_head(x.detach())
        return out


class _NoOpScaler:
    def scale(self, loss):
        return loss


class _FakePlatform:
    def empty_cache(self):
        pass


def _make_valid_hybrid_cfg():
    base = OmegaConf.load(EXPERIMENTS / "config/spatial_wovr_pi05_8gpu.yaml")
    profile = OmegaConf.load(
        EXPERIMENTS / "config/spatial_wovr_pi05_8gpu_hybrid_ppo_grpo.yaml"
    )
    cfg = OmegaConf.merge(base, profile)
    # Resolve placeholders not needed in CPU validation tests.
    cfg.cluster.component_placement = {"actor,env,rollout": "0-7"}
    cfg.runner.logger.log_path = "/tmp/hybrid_test"
    cfg.pi05_checkpoint_root = "/tmp/pi05"
    cfg.env.train.kir_frame_pool_path = "/tmp/kir"
    cfg.env.train.wan_wm_hf_ckpt_path = "/tmp/wan"
    cfg.env.train.worldpilot_root = "/tmp/wp"
    cfg.env.train.cosmos_endpoints = [
        {"host": "127.0.0.1", "port": 18059 + i} for i in range(8)
    ]
    return cfg


def test_validate_hybrid_config_accepts_preset_and_rejects_misconfigurations():
    cfg = _make_valid_hybrid_cfg()
    parsed = validate_hybrid_config(cfg, world_size=8)
    assert parsed.enabled is True
    assert parsed.rho_max == pytest.approx(0.30)
    assert parsed.pacing_mode == "linear_ramp"

    # Missing rollout value_head must fail.
    bad = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    bad.rollout.model.add_value_head = False
    with pytest.raises(ValueError, match="rollout.model.add_value_head"):
        validate_hybrid_config(bad, world_size=8)

    # Missing actor.model.openpi.add_value_head (what OpenPI factory actually reads!) must fail.
    bad = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    bad.actor.model.openpi.add_value_head = False
    with pytest.raises(ValueError, match="actor.model.openpi.add_value_head"):
        validate_hybrid_config(bad, world_size=8)

    # Missing detach_critic_input must fail.
    bad = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    bad.actor.model.openpi.detach_critic_input = False
    with pytest.raises(ValueError, match="detach_critic_input"):
        validate_hybrid_config(bad, world_size=8)

    # Missing value_lr must fail.
    bad = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    bad.actor.optim.value_lr = 0.0
    with pytest.raises(ValueError, match="value_lr"):
        validate_hybrid_config(bad, world_size=8)

    # Auto-reset on real_train breaks GAE boundaries and must fail.
    bad = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    bad.env.real_train.auto_reset = True
    with pytest.raises(ValueError, match="auto_reset=false"):
        validate_hybrid_config(bad, world_size=8)

    # Non-divisible real chunks must fail.
    bad = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    bad.algorithm.hybrid_ppo_grpo.real_micro_batch_size = 32  # 32*1*30=960, not divisible by 32*8=256
    with pytest.raises(ValueError, match="divisible"):
        validate_hybrid_config(bad, world_size=8)

    # Pure Real PPO control arm (rho_max=0.0) passes validation identically.
    baseline = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    baseline.algorithm.hybrid_ppo_grpo.rho_max = 0.0
    parsed_base = validate_hybrid_config(baseline, world_size=8)
    assert parsed_base.enabled is True
    assert parsed_base.rho_max == 0.0


def test_low_real_budget_preset_has_exact_1_to_1_optimizer_step_parity():
    cfg = _make_valid_hybrid_cfg()
    world_size = 8
    real_chunks = (
        int(cfg.env.real_train.total_num_envs)
        * int(cfg.env.real_train.rollout_epoch)
        * (int(cfg.env.real_train.max_steps_per_rollout_epoch) // 8)
    )
    wm_chunks = (
        int(cfg.env.train.total_num_envs)
        * int(cfg.env.train.rollout_epoch)
        * (int(cfg.env.train.max_steps_per_rollout_epoch) // 8)
    )
    real_steps = real_chunks // int(cfg.algorithm.hybrid_ppo_grpo.real_global_batch_size)
    wm_steps = wm_chunks // int(cfg.actor.global_batch_size)
    assert real_chunks == 960
    assert wm_chunks == 15360
    assert real_steps == wm_steps == 4
    assert int(cfg.env.real_train.total_num_envs) * int(cfg.runner.max_steps) == 512


def test_compute_wm_rho_pacing_and_safety_gates():
    cfg_ent = HybridPpoGrpoConfig(
        enabled=True,
        rho_max=0.30,
        rho_min=0.05,
        pacing_mode="entropy",
        entropy_kappa=1.0,
        entropy_power=1.0,
        min_wm_signal_groups=2,
    )
    # Critic warmup always zeros rho.
    assert compute_wm_rho(
        cfg_ent,
        in_critic_warmup=True,
        post_warmup_step=0,
        real_entropy=0.1,
        wm_signal_groups=10,
    ) == 0.0
    # Insufficient WM signal groups zeros rho (graceful fallback to pure Real PPO).
    assert compute_wm_rho(
        cfg_ent,
        in_critic_warmup=False,
        post_warmup_step=0,
        real_entropy=0.1,
        wm_signal_groups=1,
    ) == 0.0
    # Higher real entropy monotonically reduces rho, floored at rho_min.
    rho_low_h = compute_wm_rho(
        cfg_ent,
        in_critic_warmup=False,
        post_warmup_step=0,
        real_entropy=0.2,
        wm_signal_groups=5,
    )
    rho_high_h = compute_wm_rho(
        cfg_ent,
        in_critic_warmup=False,
        post_warmup_step=0,
        real_entropy=2.0,
        wm_signal_groups=5,
    )
    rho_huge_h = compute_wm_rho(
        cfg_ent,
        in_critic_warmup=False,
        post_warmup_step=0,
        real_entropy=100.0,
        wm_signal_groups=5,
    )
    assert rho_low_h == pytest.approx(0.30 / 1.2)
    assert rho_high_h == pytest.approx(0.30 / 3.0)
    assert rho_low_h > rho_high_h > rho_huge_h
    assert rho_huge_h == pytest.approx(0.05)

    # Linear ramp mode.
    cfg_ramp = HybridPpoGrpoConfig(
        enabled=True, rho_max=0.30, rho_min=0.0, pacing_mode="linear_ramp", ramp_steps=3
    )
    assert compute_wm_rho(
        cfg_ramp, in_critic_warmup=False, post_warmup_step=0, real_entropy=None, wm_signal_groups=4
    ) == pytest.approx(0.10)
    assert compute_wm_rho(
        cfg_ramp, in_critic_warmup=False, post_warmup_step=2, real_entropy=None, wm_signal_groups=4
    ) == pytest.approx(0.30)


def test_compute_paired_step_schedule_covers_all_wm_and_real_batches():
    # Equal micro-batch ratio (e.g., 15 real vs 120 WM over 15 paired steps).
    r_sched, w_sched = compute_paired_step_schedule(15, 120, target_paired_steps=15)
    assert len(r_sched) == len(w_sched) == 15
    assert [idx for step in w_sched for idx in step] == list(range(120))
    assert [idx for step in r_sched for idx in step] == list(range(15))
    assert all(len(s) == 1 for s in r_sched)
    assert all(len(s) == 8 for s in w_sched)

    # Fewer real micro-batches than paired steps wraps cleanly.
    r_sched2, w_sched2 = compute_paired_step_schedule(3, 10, target_paired_steps=5)
    assert len(r_sched2) == len(w_sched2) == 5
    assert [idx for step in w_sched2 for idx in step] == list(range(10))
    assert r_sched2 == [[0], [1], [2], [0], [1]]


def test_load_state_dict_allow_missing_value_head():
    model = _ToyActorCriticPolicy()
    grpo_state = {
        k: v.clone() for k, v in model.state_dict().items() if "value_head" not in k
    }
    missing, unexpected = load_state_dict_allow_missing_value_head(model, grpo_state)
    assert unexpected == []
    assert set(missing) == {"value_head.weight", "value_head.bias"}

    # Missing an actor key must still fail loudly.
    broken = {k: v for k, v in grpo_state.items() if k != "expert.weight"}
    with pytest.raises(RuntimeError, match="non-value_head"):
        load_state_dict_allow_missing_value_head(model, broken)

    # Unexpected key must fail loudly.
    extra = dict(grpo_state, bogus=torch.zeros(1))
    with pytest.raises(RuntimeError, match="Unexpected keys"):
        load_state_dict_allow_missing_value_head(model, extra)


def test_snapshot_weights_during_critic_warmup_captures_actor_excluding_value_head():
    model = _ToyActorCriticPolicy()
    # Mimic FSDPModelManager during critic_warmup: record trainable names, then freeze non-value_head.
    stored_names = [n for n, p in model.named_parameters() if p.requires_grad]
    for name, param in model.named_parameters():
        if "value_head" not in name:
            param.requires_grad = False

    ref = snapshot_trainable_weights(
        model, include_names=stored_names, exclude_substrings=("value_head",)
    )
    assert set(ref.keys()) == {"expert.weight", "expert.bias"}

    # Swapping still works while actor params are frozen or after unfreezing.
    orig = model.expert.weight.detach().clone()
    with torch.no_grad():
        model.expert.weight.add_(1.0)
    with swap_trainable_weights(model, ref):
        torch.testing.assert_close(model.expert.weight.detach(), orig)
    torch.testing.assert_close(model.expert.weight.detach(), orig + 1.0)


def _make_fake_actor(
    *,
    critic_warmup_steps: int,
    rho_max: float = 0.30,
    kl_beta: float = 0.01,
) -> WorldModelFSDPActor:
    torch.manual_seed(0)
    model = _ToyActorCriticPolicy()
    actor = WorldModelFSDPActor.__new__(WorldModelFSDPActor)
    actor.model = model
    actor.device = torch.device("cpu")
    actor._rank = 0
    actor._world_size = 1
    actor.version = 0
    actor.amp_context = nullcontext()
    actor.grad_scaler = _NoOpScaler()
    actor.torch_platform = _FakePlatform()
    actor.enable_sft_co_train = False
    actor._cuda_cache_reservation = None
    actor._shared_wan_owner = None
    actor._active_training_diagnostics = None
    actor.is_weight_offloaded = False
    actor.is_optimizer_offloaded = False
    actor.optimizer_steps = 0
    actor.critic_warmup_steps = critic_warmup_steps
    actor.store_requires_grad_param_name = [
        n for n, p in model.named_parameters() if p.requires_grad
    ]
    if critic_warmup_steps > 0:
        for name, param in model.named_parameters():
            if "value_head" not in name:
                param.requires_grad = False

    def _build_opt():
        actor_params = [
            p for n, p in model.named_parameters() if p.requires_grad and "value_head" not in n
        ]
        critic_params = [
            p for n, p in model.named_parameters() if p.requires_grad and "value_head" in n
        ]
        groups = []
        if actor_params:
            groups.append({"params": actor_params, "lr": 1e-2})
        if critic_params:
            groups.append({"params": critic_params, "lr": 5e-2})
        return torch.optim.AdamW(groups)

    actor.optimizer = _build_opt()
    actor.lr_scheduler = SimpleNamespace(step=lambda: None)
    actor.is_last_history = []
    actor.before_micro_batch = lambda m, is_last_micro_batch: (
        actor.is_last_history.append(bool(is_last_micro_batch)) or nullcontext()
    )

    def _optimizer_step():
        actor.optimizer.step()
        actor.optimizer_steps += 1
        lrs = [g["lr"] for g in actor.optimizer.param_groups]
        if actor.critic_warmup_steps > 0 and actor.optimizer_steps >= actor.critic_warmup_steps:
            for name, param in model.named_parameters():
                if name in actor.store_requires_grad_param_name:
                    param.requires_grad = True
            actor.critic_warmup_steps = 0
            actor.optimizer = _build_opt()
        return 1.0, lrs

    actor.optimizer_step = _optimizer_step
    actor.cfg = OmegaConf.create(
        {
            "runner": {"task_type": "embodied", "resume_dir": None},
            "actor": {
                "seed": 42,
                "micro_batch_size": 2,
                "global_batch_size": 4,
                "model": {
                    "model_type": "openpi",
                    "action_dim": 7,
                    "num_action_chunks": 8,
                    "add_value_head": True,
                },
            },
            "env": {
                "train": {"max_episode_steps": 16},
                "real_train": {"max_episode_steps": 16},
            },
            "algorithm": {
                "adv_type": "grpo",
                "loss_type": "actor",
                "reward_type": "action_level",
                "logprob_type": "token_level",
                "entropy_type": "token_level",
                "entropy_bonus": 0.0,
                "clip_ratio_high": 0.28,
                "clip_ratio_low": 0.2,
                "kl_penalty": "low_var_kl",
                "kl_beta": kl_beta,
                "update_epoch": 1,
                "hybrid_ppo_grpo": {
                    "enabled": True,
                    "rho_max": rho_max,
                    "rho_min": 0.0,
                    "pacing_mode": "entropy",
                    "entropy_kappa": 0.5,
                    "entropy_power": 1.0,
                    "real_global_batch_size": 4,
                    "skip_wm_during_critic_warmup": True,
                    "min_wm_signal_groups": 1,
                },
            },
        }
    )
    actor._kl_beta = kl_beta
    actor._kl_penalty_type = "low_var_kl"
    actor._hybrid_cfg = parse_hybrid_config(actor.cfg.algorithm)
    actor._ref_trainable_weights = actor._snapshot_policy_reference_weights()
    actor._hybrid_wm_skipped = False
    actor._hybrid_effective_rho = 0.0
    actor._hybrid_last_real_entropy = None
    actor._hybrid_post_warmup_steps = 0
    actor.real_rollout_batch = None
    actor.rollout_batch = None
    return actor


def _make_real_batch(model: _ToyActorCriticPolicy, T: int = 2, B: int = 2) -> dict:
    obs = torch.randn(T, B, 4)
    target = torch.randn(T, B, 8, 7)
    with torch.no_grad():
        out = model(
            {"obs": obs.reshape(-1, 4), "target": target.reshape(-1, 8, 7)},
            compute_logprobs=True,
            compute_values=True,
        )
    prev_lp = out["logprobs"].reshape(T, B, 8, 7).detach()
    prev_v = out["values"].reshape(T, B, 1).detach()
    prev_v_full = torch.cat([prev_v, torch.zeros(1, B, 1)], dim=0)
    rewards = torch.zeros(T, B, 8)
    rewards[-1, 0, -1] = 1.0
    dones = torch.zeros(T + 1, B, 8, dtype=torch.bool)
    dones[-1] = True
    return {
        "forward_inputs": {"obs": obs, "target": target},
        "prev_logprobs": prev_lp,
        "prev_values": prev_v_full,
        "rewards": rewards,
        "dones": dones,
        "loss_mask": torch.ones(T, B, 1, dtype=torch.bool),
        "loss_mask_sum": torch.full((T, B, 1), float(T)),
    }


def _make_wm_batch(model: _ToyActorCriticPolicy, T: int = 2, B: int = 4) -> dict:
    obs = torch.randn(T, B, 4)
    target = torch.randn(T, B, 8, 7)
    with torch.no_grad():
        out = model(
            {"obs": obs.reshape(-1, 4), "target": target.reshape(-1, 8, 7)},
            compute_logprobs=True,
            compute_values=False,
        )
    prev_lp = out["logprobs"].reshape(T, B, 8, 7).detach()
    adv = torch.tensor([1.0, -1.0, 0.5, -0.5]).reshape(1, B, 1).expand(T, B, 8).clone()
    return {
        "forward_inputs": {"obs": obs, "target": target},
        "prev_logprobs": prev_lp,
        "advantages": adv,
        "loss_mask": torch.ones(T, B, 8, dtype=torch.bool),
        "loss_mask_sum": torch.full((T, B, 8), float(T * 8)),
    }


def test_hybrid_critic_warmup_trains_only_value_head_and_skips_wm():
    actor = _make_fake_actor(critic_warmup_steps=1, rho_max=0.30, kl_beta=0.01)
    assert actor.should_run_wm_stream() is False

    actor_w_before = actor.model.expert.weight.detach().clone()
    critic_w_before = actor.model.value_head.weight.detach().clone()

    actor.real_rollout_batch = _make_real_batch(actor.model, T=2, B=2)
    actor.compute_real_advantages_and_returns()
    skip_metrics = actor.skip_wm_stream()
    assert skip_metrics["hybrid/wm_skipped"] == 1.0

    train_metrics = actor.run_training()
    assert train_metrics["hybrid/in_critic_warmup"] == 1.0
    assert train_metrics["hybrid/rho"] == 0.0
    assert "critic/value_loss" in train_metrics

    # Actor weights must stay untouched during critic warmup; value_head must update.
    torch.testing.assert_close(actor.model.expert.weight.detach(), actor_w_before)
    assert not torch.allclose(actor.model.value_head.weight.detach(), critic_w_before)
    # Warmup complete -> WM stream is now enabled and actor is unfrozen.
    assert actor.should_run_wm_stream() is True
    assert actor.model.expert.weight.requires_grad is True


def test_hybrid_joint_update_isolates_value_head_from_wm_and_syncs_once_per_paired_step():
    actor = _make_fake_actor(critic_warmup_steps=0, rho_max=0.30, kl_beta=0.02)
    assert actor.should_run_wm_stream() is True

    # First verify WM micro-batch alone puts ZERO gradient on value_head.
    wm_batch = _make_wm_batch(actor.model, T=2, B=4)
    actor.optimizer.zero_grad()
    wm_micro = {
        "forward_inputs": {
            "obs": wm_batch["forward_inputs"]["obs"][0, :2],
            "target": wm_batch["forward_inputs"]["target"][0, :2],
        },
        "prev_logprobs": wm_batch["prev_logprobs"][0, :2],
        "advantages": wm_batch["advantages"][0, :2],
        "loss_mask": wm_batch["loss_mask"][0, :2],
        "loss_mask_sum": wm_batch["loss_mask_sum"][0, :2],
    }
    actor._train_wm_grpo_micro_batch(wm_micro, {}, loss_scale=0.3, is_last=True)
    assert actor.model.expert.weight.grad is not None
    assert actor.model.value_head.weight.grad is None

    # Now run a full paired Real-PPO + WM-GRPO step (real_size=4 -> 2 mbs, wm_size=8 -> 4 mbs, 2 paired steps).
    actor.optimizer.zero_grad()
    actor.is_last_history.clear()
    actor.real_rollout_batch = _make_real_batch(actor.model, T=2, B=2)
    actor.compute_real_advantages_and_returns()
    actor.rollout_batch = _make_wm_batch(actor.model, T=2, B=4)
    actor._hybrid_wm_skipped = False
    actor._hybrid_effective_rho = 0.25

    actor_w_before = actor.model.expert.weight.detach().clone()
    critic_w_before = actor.model.value_head.weight.detach().clone()

    metrics = actor.run_training()
    assert metrics["hybrid/rho"] == pytest.approx(0.25)
    assert "real/policy_loss" in metrics
    assert "wm/policy_loss" in metrics
    assert "critic/value_loss" in metrics
    assert "real/kl_loss" in metrics
    assert "wm/kl_loss" in metrics
    assert "hybrid/real_entropy" in metrics
    assert actor._hybrid_last_real_entropy is not None

    # Both actor and critic updated.
    assert not torch.allclose(actor.model.expert.weight.detach(), actor_w_before)
    assert not torch.allclose(actor.model.value_head.weight.detach(), critic_w_before)
    # New gradient fusion diagnostics are logged.
    assert "hybrid/grad_cos_sim" in metrics
    assert "hybrid/grad_real_norm" in metrics
    assert "hybrid/grad_wm_norm" in metrics
    assert "hybrid/wm_grad_accepted" in metrics
    # 2 paired steps, each with 1 real mb (syncs at end of real phase) + 2 wm mbs (syncs at end of WM phase).
    assert actor.is_last_history == [True, False, True, True, False, True]


def test_merge_real_and_wm_gradients_norm_match_cosine_gate_and_pcgrad():
    model = _ToyActorCriticPolicy()
    # Set exact known real actor + critic gradients.
    g_real_actor_w = torch.zeros_like(model.expert.weight)
    g_real_actor_w[0, 0] = 3.0
    g_real_actor_w[0, 1] = 4.0  # ||g_real|| = 5.0
    g_real_actor_b = torch.zeros_like(model.expert.bias)
    g_real_critic_w = torch.full_like(model.value_head.weight, 2.5)
    g_real_critic_b = torch.full_like(model.value_head.bias, -1.25)

    model.expert.weight.grad = g_real_actor_w.clone()
    model.expert.bias.grad = g_real_actor_b.clone()
    model.value_head.weight.grad = g_real_critic_w.clone()
    model.value_head.bias.grad = g_real_critic_b.clone()
    real_actor_cpu, real_critic_cpu = snapshot_param_grads_to_cpu(model)

    # Case 1: Aligned WM gradient with 100x smaller raw scale (like token_level vs chunk_level).
    model.zero_grad()
    g_wm_aligned = torch.zeros_like(model.expert.weight)
    g_wm_aligned[0, 0] = 0.03
    g_wm_aligned[0, 1] = 0.04  # exact same direction, 100x smaller norm = 0.05
    model.expert.weight.grad = g_wm_aligned.clone()
    model.expert.bias.grad = torch.zeros_like(model.expert.bias)
    m_aligned = merge_real_and_wm_gradients(
        model,
        real_actor_cpu,
        real_critic_cpu,
        rho=0.30,
        grad_norm_match=True,
        grad_conflict_mode="cosine_gate",
        grad_cos_min=0.0,
    )
    assert m_aligned["hybrid/grad_cos_sim"] == pytest.approx(1.0, abs=1e-5)
    assert m_aligned["hybrid/grad_norm_ratio"] == pytest.approx(0.01, abs=1e-5)
    assert m_aligned["hybrid/wm_grad_accepted"] == 1.0
    # After norm-matching, g_total == (1 + 0.30) * g_real regardless of 100x raw scale gap.
    torch.testing.assert_close(model.expert.weight.grad, 1.30 * g_real_actor_w)
    # Value head grads restored verbatim.
    torch.testing.assert_close(model.value_head.weight.grad, g_real_critic_w)
    torch.testing.assert_close(model.value_head.bias.grad, g_real_critic_b)

    # Case 2: Conflicting WM gradient in cosine_gate mode -> dropped completely (g_total == g_real).
    model.zero_grad()
    g_wm_conflict = torch.zeros_like(model.expert.weight)
    g_wm_conflict[0, 0] = -3.0
    g_wm_conflict[0, 1] = 1.0  # dot = -9 + 4 = -5 < 0
    model.expert.weight.grad = g_wm_conflict.clone()
    model.expert.bias.grad = torch.zeros_like(model.expert.bias)
    m_gate = merge_real_and_wm_gradients(
        model,
        real_actor_cpu,
        real_critic_cpu,
        rho=0.30,
        grad_norm_match=True,
        grad_conflict_mode="cosine_gate",
        grad_cos_min=0.0,
    )
    assert m_gate["hybrid/grad_cos_sim"] < 0.0
    assert m_gate["hybrid/wm_grad_accepted"] == 0.0
    torch.testing.assert_close(model.expert.weight.grad, g_real_actor_w)
    torch.testing.assert_close(model.value_head.weight.grad, g_real_critic_w)

    # Case 3: Conflicting WM gradient in pcgrad mode -> <g_total, g_real> == ||g_real||^2 exactly.
    model.zero_grad()
    model.expert.weight.grad = g_wm_conflict.clone()
    model.expert.bias.grad = torch.zeros_like(model.expert.bias)
    m_pc = merge_real_and_wm_gradients(
        model,
        real_actor_cpu,
        real_critic_cpu,
        rho=0.30,
        grad_norm_match=True,
        grad_conflict_mode="pcgrad",
        grad_cos_min=0.0,
    )
    assert m_pc["hybrid/wm_grad_projected"] == 1.0
    dot_after = float(torch.sum(model.expert.weight.grad * g_real_actor_w).item())
    real_sq = float(torch.sum(g_real_actor_w * g_real_actor_w).item())
    assert dot_after == pytest.approx(real_sq, rel=1e-5)
    # Opposite WM gradient (cos = -1) projects to exact g_real.
    model.zero_grad()
    model.expert.weight.grad = -2.0 * g_real_actor_w
    model.expert.bias.grad = torch.zeros_like(model.expert.bias)
    merge_real_and_wm_gradients(
        model,
        real_actor_cpu,
        real_critic_cpu,
        rho=0.30,
        grad_norm_match=True,
        grad_conflict_mode="pcgrad",
        grad_cos_min=0.0,
    )
    torch.testing.assert_close(model.expert.weight.grad, g_real_actor_w)
    torch.testing.assert_close(model.value_head.weight.grad, g_real_critic_w)


def test_install_value_head_checkpoint_compat_loads_actor_only_native_checkpoint(tmp_path):
    import rlinf.models.embodiment.openpi as openpi_pkg

    model = _ToyActorCriticPolicy()
    model.config = SimpleNamespace(add_value_head=True)
    actor_only_sd = {
        k: v.clone() for k, v in model.state_dict().items() if not k.startswith("value_head.")
    }
    ckpt_dir = tmp_path / "actor_only"
    (ckpt_dir / "actor/model_state_dict").mkdir(parents=True)
    torch.save(actor_only_sd, ckpt_dir / "actor/model_state_dict/full_weights.pt")

    install_value_head_checkpoint_compat(seed=42)
    openpi_pkg._load_state_dict(model, str(ckpt_dir), "native")
    torch.testing.assert_close(model.expert.weight, actor_only_sd["expert.weight"])

    # Still rejects genuinely broken checkpoints missing actor keys.
    broken_sd = {k: v for k, v in actor_only_sd.items() if k != "expert.weight"}
    broken_dir = tmp_path / "broken"
    broken_dir.mkdir(parents=True)
    torch.save(broken_sd, broken_dir / "full_weights.pt")
    with pytest.raises(RuntimeError, match="incompatible"):
        openpi_pkg._load_state_dict(model, str(broken_dir), "native")


def test_hybrid_runner_orchestrates_warmup_skip_and_post_warmup_streams():
    events: list[str] = []

    class _Handle:
        def __init__(self, tag, value):
            self.tag = tag
            self.value = value

        def wait(self):
            events.append(f"wait:{self.tag}")
            return self.value

    class _FakeActorProxy:
        def __init__(self):
            self.step = 0

        def set_global_step(self, step):
            self.step = step
            events.append(f"actor.set_step:{step}")

        def recv_real_rollout_trajectories(self, input_channel):
            events.append("actor.recv_real")
            return _Handle("actor.recv_real", None)

        def compute_real_advantages_and_returns(self):
            events.append("actor.adv_real")
            return _Handle("actor.adv_real", [{"real_ret": 1.0}])

        def should_run_wm_stream(self):
            # Step 0 is critic warmup (skip WM); Step 1 runs WM.
            return _Handle("actor.should_wm", [self.step >= 1])

        def skip_wm_stream(self):
            events.append("actor.skip_wm")
            return _Handle("actor.skip_wm", [{"hybrid/wm_skipped": 1.0}])

        def recv_rollout_trajectories(self, input_channel):
            events.append("actor.recv_wm")
            return _Handle("actor.recv_wm", None)

        def compute_advantages_and_returns(self):
            events.append("actor.adv_wm")
            return _Handle("actor.adv_wm", [{"hybrid/rho": 0.25}])

        def run_training(self):
            events.append("actor.train")
            return _Handle("actor.train", [{"train_loss": 0.5}])

    class _FakeEnvProxy:
        def interact_real(self, **kwargs):
            events.append("env.interact_real")
            return _Handle("env.real", [None])

        def interact(self, **kwargs):
            events.append("env.interact_wm")
            return _Handle("env.wm", [None])

    class _FakeRolloutProxy:
        def set_global_step(self, step):
            pass

        def generate_real(self, **kwargs):
            events.append("rollout.gen_real")
            return _Handle("rollout.real", None)

        def generate(self, **kwargs):
            events.append("rollout.gen_wm")
            return _Handle("rollout.wm", None)

    class _DummyTimer:
        def __call__(self, name, **kwargs):
            return nullcontext()

        def consume_durations(self):
            return {}

    class _FakeRunner(HybridRunnerMixin):
        def __init__(self):
            self.cfg = OmegaConf.create({"algorithm": {"hybrid_ppo_grpo": {"enabled": True}}})
            self.global_step = 0
            self.max_steps = 2
            self.actor = _FakeActorProxy()
            self.env = _FakeEnvProxy()
            self.rollout = _FakeRolloutProxy()
            self.env_channel = None
            self.rollout_channel = None
            self.reward_channel = None
            self.actor_channel = None
            self.timer = _DummyTimer()
            self.metric_logger = SimpleNamespace(log=lambda data, step: None)

        def _should_sync_rollout_weights(self, step):
            return False

        def _maybe_eval_and_checkpoint(self, step):
            return {}

        def _aggregate_numeric_metrics(self, items):
            return items[0] if items else {}

        def print_metrics_table_async(self, *args, **kwargs):
            pass

        def _finish_run(self):
            events.append("finish")

    runner = _FakeRunner()
    runner.run()

    # Step 0 skipped WM; Step 1 waited for env.wm offload BEFORE actor.adv_wm & actor.train.
    assert "actor.skip_wm" in events
    assert events.index("wait:env.wm") < events.index("actor.adv_wm") < events.index("actor.train") or (
        events.index("wait:env.wm") < events.index("actor.adv_wm")
    )
    assert events.count("env.interact_real") == 2
    assert events.count("env.interact_wm") == 1
    assert events[-1] == "finish"
