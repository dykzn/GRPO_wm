"""WorldSample Route-A Hybrid: Real-Env PPO + Closed-Loop World-Model GRPO.

Architecture
------------
Each outer iteration runs two strictly isolated rollout/advantage streams on
the same colocated actor/rollout/env cluster:

1. **Real stream (LIBERO simulator -> PPO + GAE)**:
   - Trains both the policy and ``value_head`` using unbiased simulator rewards.
   - During ``critic_warmup_steps``, only the real stream runs (skipping Wan +
     Cosmos entirely) so the randomly initialized ``value_head`` converges cleanly
     before joint policy updates begin.
2. **Synthetic stream (Wan + Cosmos wrist + ResNet RM -> GRPO)**:
   - Closed-loop rollouts in the world model with independent group-relative
     GRPO normalization, reward filtering, and optional Tier-2 hallucination
     trust scaling.
   - Never feeds into ``value_head`` (``compute_values=False``), isolating the
     critic from world-model reward bias.
   - Weighted by WorldSample-style pacing ``rho(H_t)`` (default entropy-paced,
     capped at ``rho_max <= 0.30``) and gated to ``0.0`` whenever fewer than
     ``min_wm_signal_groups`` mixed-outcome WM groups survive filtering.

Joint gradient per paired optimizer step:
    L = L_PPO_actor_critic(real) + beta_real * L_KL(real)
        + rho(H_t) * (L_GRPO_actor(wm) + beta_wm * L_KL(wm))
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

from omegaconf import OmegaConf
import torch


SUPPORTED_PACING_MODES = frozenset({"constant", "entropy", "linear_ramp"})
SUPPORTED_GRAD_CONFLICT_MODES = frozenset({"cosine_gate", "pcgrad", "none"})


@dataclass(frozen=True)
class HybridPpoGrpoConfig:
    enabled: bool = False
    rho_max: float = 0.30
    rho_min: float = 0.0
    pacing_mode: str = "entropy"
    entropy_kappa: float = 0.5
    entropy_power: float = 1.0
    entropy_ref: float = 0.0
    ramp_steps: int = 5
    real_clip_ratio_low: float = 0.2
    real_clip_ratio_high: float = 0.2
    real_value_clip: float = 0.2
    real_huber_delta: float = 10.0
    real_value_loss_coef: float = 1.0
    real_kl_beta: float | None = None
    wm_kl_beta: float | None = None
    real_micro_batch_size: int | None = None
    real_global_batch_size: int | None = None
    paired_steps_per_epoch: int | None = None
    real_update_epochs: int = 1
    min_wm_signal_groups: int = 1
    skip_wm_during_critic_warmup: bool = True
    grad_norm_match: bool = True
    grad_conflict_mode: str = "cosine_gate"
    grad_cos_min: float = 0.0


def _as_float(value: Any, name: str, *, min_value: float | None = None) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if min_value is not None and result < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {result}")
    return result


def _as_int(value: Any, name: str, *, min_value: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if min_value is not None and value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value}")
    return value


def parse_hybrid_config(algorithm_cfg: Any) -> HybridPpoGrpoConfig:
    """Parse ``algorithm.hybrid_ppo_grpo`` into a validated dataclass."""
    raw = (
        algorithm_cfg.get("hybrid_ppo_grpo", None)
        if algorithm_cfg is not None
        else None
    )
    if raw is None or not bool(raw.get("enabled", False)):
        return HybridPpoGrpoConfig(enabled=False)

    rho_max = _as_float(raw.get("rho_max", 0.30), "hybrid_ppo_grpo.rho_max", min_value=0.0)
    rho_min = _as_float(raw.get("rho_min", 0.0), "hybrid_ppo_grpo.rho_min", min_value=0.0)
    if rho_min > rho_max:
        raise ValueError(
            f"hybrid_ppo_grpo.rho_min ({rho_min}) cannot exceed rho_max ({rho_max})"
        )
    pacing_mode = str(raw.get("pacing_mode", "entropy"))
    if pacing_mode not in SUPPORTED_PACING_MODES:
        raise ValueError(
            f"hybrid_ppo_grpo.pacing_mode must be one of {sorted(SUPPORTED_PACING_MODES)}, "
            f"got {pacing_mode!r}"
        )
    entropy_kappa = _as_float(
        raw.get("entropy_kappa", 0.5), "hybrid_ppo_grpo.entropy_kappa", min_value=0.0
    )
    entropy_power = _as_float(
        raw.get("entropy_power", 1.0), "hybrid_ppo_grpo.entropy_power", min_value=0.0
    )
    entropy_ref = _as_float(
        raw.get("entropy_ref", 0.0), "hybrid_ppo_grpo.entropy_ref"
    )
    ramp_steps = _as_int(
        raw.get("ramp_steps", 5), "hybrid_ppo_grpo.ramp_steps", min_value=1
    )
    real_clip_low = _as_float(
        raw.get("real_clip_ratio_low", 0.2),
        "hybrid_ppo_grpo.real_clip_ratio_low",
        min_value=0.0,
    )
    real_clip_high = _as_float(
        raw.get("real_clip_ratio_high", 0.2),
        "hybrid_ppo_grpo.real_clip_ratio_high",
        min_value=0.0,
    )
    real_value_clip = _as_float(
        raw.get("real_value_clip", 0.2),
        "hybrid_ppo_grpo.real_value_clip",
        min_value=0.0,
    )
    real_huber_delta = _as_float(
        raw.get("real_huber_delta", 10.0),
        "hybrid_ppo_grpo.real_huber_delta",
        min_value=1e-6,
    )
    real_value_loss_coef = _as_float(
        raw.get("real_value_loss_coef", 1.0),
        "hybrid_ppo_grpo.real_value_loss_coef",
        min_value=0.0,
    )
    raw_real_kl = raw.get("real_kl_beta", None)
    real_kl_beta = (
        None
        if raw_real_kl is None
        else _as_float(raw_real_kl, "hybrid_ppo_grpo.real_kl_beta", min_value=0.0)
    )
    raw_wm_kl = raw.get("wm_kl_beta", None)
    wm_kl_beta = (
        None
        if raw_wm_kl is None
        else _as_float(raw_wm_kl, "hybrid_ppo_grpo.wm_kl_beta", min_value=0.0)
    )
    raw_real_mbs = raw.get("real_micro_batch_size", None)
    real_micro_batch_size = (
        None
        if raw_real_mbs is None
        else _as_int(raw_real_mbs, "hybrid_ppo_grpo.real_micro_batch_size", min_value=1)
    )
    raw_real_gbs = raw.get("real_global_batch_size", None)
    real_global_batch_size = (
        None
        if raw_real_gbs is None
        else _as_int(raw_real_gbs, "hybrid_ppo_grpo.real_global_batch_size", min_value=1)
    )
    raw_paired_steps = raw.get("paired_steps_per_epoch", None)
    paired_steps_per_epoch = (
        None
        if raw_paired_steps is None
        else _as_int(
            raw_paired_steps, "hybrid_ppo_grpo.paired_steps_per_epoch", min_value=1
        )
    )
    real_update_epochs = _as_int(
        raw.get("real_update_epochs", 1),
        "hybrid_ppo_grpo.real_update_epochs",
        min_value=1,
    )
    min_wm_signal_groups = _as_int(
        raw.get("min_wm_signal_groups", 1),
        "hybrid_ppo_grpo.min_wm_signal_groups",
        min_value=0,
    )
    skip_wm_during_critic_warmup = bool(
        raw.get("skip_wm_during_critic_warmup", True)
    )
    grad_norm_match = bool(raw.get("grad_norm_match", True))
    grad_conflict_mode = str(raw.get("grad_conflict_mode", "cosine_gate"))
    if grad_conflict_mode not in SUPPORTED_GRAD_CONFLICT_MODES:
        raise ValueError(
            f"hybrid_ppo_grpo.grad_conflict_mode must be one of "
            f"{sorted(SUPPORTED_GRAD_CONFLICT_MODES)}, got {grad_conflict_mode!r}"
        )
    grad_cos_min = _as_float(
        raw.get("grad_cos_min", 0.0), "hybrid_ppo_grpo.grad_cos_min"
    )
    if not (-1.0 <= grad_cos_min <= 1.0):
        raise ValueError(
            f"hybrid_ppo_grpo.grad_cos_min must be in [-1, 1], got {grad_cos_min}"
        )

    return HybridPpoGrpoConfig(
        enabled=True,
        rho_max=rho_max,
        rho_min=rho_min,
        pacing_mode=pacing_mode,
        entropy_kappa=entropy_kappa,
        entropy_power=entropy_power,
        entropy_ref=entropy_ref,
        ramp_steps=ramp_steps,
        real_clip_ratio_low=real_clip_low,
        real_clip_ratio_high=real_clip_high,
        real_value_clip=real_value_clip,
        real_huber_delta=real_huber_delta,
        real_value_loss_coef=real_value_loss_coef,
        real_kl_beta=real_kl_beta,
        wm_kl_beta=wm_kl_beta,
        real_micro_batch_size=real_micro_batch_size,
        real_global_batch_size=real_global_batch_size,
        paired_steps_per_epoch=paired_steps_per_epoch,
        real_update_epochs=real_update_epochs,
        min_wm_signal_groups=min_wm_signal_groups,
        skip_wm_during_critic_warmup=skip_wm_during_critic_warmup,
        grad_norm_match=grad_norm_match,
        grad_conflict_mode=grad_conflict_mode,
        grad_cos_min=grad_cos_min,
    )


def validate_hybrid_config(cfg: Any, *, world_size: int) -> HybridPpoGrpoConfig:
    """Validate full Hydra config when hybrid Real-PPO + WM-GRPO is enabled."""
    hybrid = parse_hybrid_config(cfg.get("algorithm", None))
    if not hybrid.enabled:
        return hybrid

    for role in ("actor", "rollout"):
        model_cfg = cfg[role].model
        openpi_cfg = model_cfg.get("openpi", {})
        if not bool(model_cfg.get("add_value_head", False)):
            raise ValueError(
                f"Hybrid PPO+GRPO requires {role}.model.add_value_head=true"
            )
        if not bool(openpi_cfg.get("add_value_head", False)):
            raise ValueError(
                f"Hybrid PPO+GRPO requires {role}.model.openpi.add_value_head=true "
                "(OpenPI's model factory reads openpi.add_value_head)"
            )
        if not bool(openpi_cfg.get("value_after_vlm", False)):
            raise ValueError(
                f"Hybrid PPO+GRPO requires {role}.model.openpi.value_after_vlm=true for Pi0.5"
            )
        if not bool(openpi_cfg.get("detach_critic_input", False)):
            raise ValueError(
                f"Hybrid PPO+GRPO requires {role}.model.openpi.detach_critic_input=true "
                "to prevent critic gradients from corrupting frozen/shared VLM representations"
            )

    value_lr = cfg.actor.optim.get("value_lr", None)
    if value_lr is None or float(value_lr) <= 0.0:
        raise ValueError("Hybrid PPO+GRPO requires actor.optim.value_lr > 0")

    real_env = cfg.env.get("real_train", None)
    if real_env is None:
        raise ValueError("Hybrid PPO+GRPO requires an env.real_train block")
    if real_env.get("env_type", None) != "libero":
        raise ValueError("Hybrid PPO+GRPO requires env.real_train.env_type='libero'")
    if real_env.get("task_suite_name", None) != "libero_spatial":
        raise ValueError(
            "Spatial Hybrid PPO+GRPO requires env.real_train.task_suite_name='libero_spatial'"
        )
    if bool(real_env.get("auto_reset", False)) or bool(
        real_env.get("ignore_terminations", False)
    ):
        raise ValueError(
            "Hybrid PPO+GRPO requires env.real_train.auto_reset=false and "
            "ignore_terminations=false for clean GAE episode boundaries"
        )
    if int(real_env.total_num_envs) % world_size != 0 or int(real_env.total_num_envs) < world_size:
        raise ValueError(
            f"env.real_train.total_num_envs ({real_env.total_num_envs}) must be a "
            f"positive multiple of world_size ({world_size})"
        )
    chunk_size = int(cfg.actor.model.num_action_chunks)
    if int(real_env.max_steps_per_rollout_epoch) % chunk_size != 0:
        raise ValueError(
            f"env.real_train.max_steps_per_rollout_epoch ({real_env.max_steps_per_rollout_epoch}) "
            f"must be divisible by num_action_chunks ({chunk_size})"
        )

    real_chunks = (
        int(real_env.total_num_envs)
        * int(real_env.rollout_epoch)
        * (int(real_env.max_steps_per_rollout_epoch) // chunk_size)
    )
    real_mbs = (
        int(hybrid.real_micro_batch_size)
        if hybrid.real_micro_batch_size is not None
        else int(cfg.actor.micro_batch_size)
    )
    micro_global = real_mbs * world_size
    if real_chunks % micro_global != 0:
        raise ValueError(
            f"Real PPO rollout chunks ({real_chunks}) must be divisible by "
            f"real_micro_batch_size * world_size ({micro_global})"
        )
    if hybrid.real_global_batch_size is not None:
        if hybrid.real_global_batch_size % micro_global != 0:
            raise ValueError(
                f"hybrid_ppo_grpo.real_global_batch_size ({hybrid.real_global_batch_size}) "
                f"must be divisible by real_micro_batch_size * world_size ({micro_global})"
            )
        if real_chunks % hybrid.real_global_batch_size != 0:
            raise ValueError(
                f"Real PPO rollout chunks ({real_chunks}) must be divisible by "
                f"hybrid_ppo_grpo.real_global_batch_size ({hybrid.real_global_batch_size})"
            )
    if real_env.get("openpi_gripper_action_mode", None) != "libero_01_to_signed":
        raise ValueError(
            "Spatial Hybrid PPO+GRPO requires env.real_train.openpi_gripper_action_mode='libero_01_to_signed'"
        )
    return hybrid


def compute_wm_rho(
    cfg: HybridPpoGrpoConfig,
    *,
    in_critic_warmup: bool,
    post_warmup_step: int,
    real_entropy: float | None,
    wm_signal_groups: float | None,
) -> float:
    """Compute the WorldSample pacing coefficient ``rho(H_t)`` for the WM stream."""
    if not cfg.enabled or in_critic_warmup or cfg.rho_max <= 0.0:
        return 0.0
    if wm_signal_groups is not None and wm_signal_groups < float(
        cfg.min_wm_signal_groups
    ):
        return 0.0

    if cfg.pacing_mode == "constant":
        rho = cfg.rho_max
    elif cfg.pacing_mode == "linear_ramp":
        frac = min(1.0, float(max(0, post_warmup_step) + 1) / float(cfg.ramp_steps))
        rho = cfg.rho_min + frac * (cfg.rho_max - cfg.rho_min)
    elif cfg.pacing_mode == "entropy":
        if real_entropy is None or not math.isfinite(real_entropy):
            rho = cfg.rho_max
        else:
            excess = max(0.0, float(real_entropy) - cfg.entropy_ref)
            denom = 1.0 + cfg.entropy_kappa * (excess**cfg.entropy_power)
            rho = cfg.rho_max / denom
    else:
        raise ValueError(f"Unsupported pacing_mode: {cfg.pacing_mode!r}")

    return float(min(cfg.rho_max, max(cfg.rho_min, rho)))


def compute_paired_step_schedule(
    real_num_micro_batches: int,
    wm_num_micro_batches: int,
    *,
    target_paired_steps: int,
    real_update_epochs: int = 1,
) -> tuple[list[list[int]], list[list[int]]]:
    """Build matched ``(real_indices, wm_indices)`` lists for each paired optimizer step.

    Guarantees:
    - Every paired step receives at least 1 real micro-batch and at least 1 WM
      micro-batch (assuming both inputs > 0).
    - Every WM micro-batch is consumed exactly once across the paired steps.
    - When ``real_num_micro_batches * real_update_epochs >= target_paired_steps``,
      real micro-batches are partitioned across paired steps with no dropped
      samples; when fewer are available, they wrap cleanly so every paired step
      remains anchored by real PPO gradients.
    """
    if real_num_micro_batches <= 0 or wm_num_micro_batches <= 0:
        raise ValueError("Both real and WM micro-batch counts must be positive")
    if target_paired_steps <= 0:
        raise ValueError("target_paired_steps must be positive")
    if wm_num_micro_batches < target_paired_steps:
        target_paired_steps = wm_num_micro_batches

    # Partition WM micro-batches evenly across target_paired_steps.
    wm_schedule: list[list[int]] = []
    wm_cursor = 0
    for step_idx in range(target_paired_steps):
        remaining_steps = target_paired_steps - step_idx
        remaining_wm = wm_num_micro_batches - wm_cursor
        take = (remaining_wm + remaining_steps - 1) // remaining_steps
        wm_schedule.append(list(range(wm_cursor, wm_cursor + take)))
        wm_cursor += take

    # Build real index stream (repeat real_update_epochs times, then wrap if needed).
    total_real_pool = [
        idx % real_num_micro_batches
        for idx in range(real_num_micro_batches * max(1, real_update_epochs))
    ]
    real_schedule: list[list[int]] = []
    if len(total_real_pool) >= target_paired_steps:
        real_cursor = 0
        for step_idx in range(target_paired_steps):
            remaining_steps = target_paired_steps - step_idx
            remaining_real = len(total_real_pool) - real_cursor
            take = (remaining_real + remaining_steps - 1) // remaining_steps
            real_schedule.append(total_real_pool[real_cursor : real_cursor + take])
            real_cursor += take
    else:
        for step_idx in range(target_paired_steps):
            real_schedule.append([total_real_pool[step_idx % len(total_real_pool)]])

    return real_schedule, wm_schedule


def load_state_dict_allow_missing_value_head(
    model: torch.nn.Module, state_dict: dict[str, torch.Tensor]
) -> tuple[list[str], list[str]]:
    """Load a GRPO/SFT checkpoint into a PPO model allowing only ``value_head`` to be missing."""
    result = model.load_state_dict(state_dict, strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)
    if unexpected:
        raise RuntimeError(
            f"Unexpected keys when loading checkpoint into hybrid PPO+GRPO model: {unexpected[:10]}"
        )
    non_value_missing = [
        key for key in missing if "value_head" not in key.split(".")
    ]
    if non_value_missing:
        raise RuntimeError(
            "Missing non-value_head keys when loading checkpoint into hybrid PPO+GRPO model: "
            f"{non_value_missing[:10]}"
        )
    return missing, unexpected


def install_value_head_checkpoint_compat(*, seed: int | None = None) -> None:
    """Allow native OpenPI loader to initialize a fresh ``value_head`` on GRPO/SFT weights.

    RLinf's ``rlinf.models.embodiment.openpi._load_state_dict`` rejects missing
    ``value_head.*`` keys even when ``add_value_head=True`` is requested on an
    actor-only checkpoint (such as ``pi05/global_step_12000``). Patching inside
    each Ray worker process avoids duplicating 16 GB checkpoints on disk while
    preserving strict checks on every non-``value_head`` tensor.
    """
    import rlinf.models.embodiment.openpi as openpi_pkg

    if getattr(openpi_pkg._load_state_dict, "_hybrid_value_head_compat", False):
        return
    original_loader = openpi_pkg._load_state_dict

    def _compat_load_state_dict(
        model: torch.nn.Module,
        checkpoint_dir: str,
        checkpoint_format: str,
        checkpoint_variant: str = "fp32",
    ):
        if not bool(getattr(getattr(model, "config", None), "add_value_head", False)):
            return original_loader(
                model, checkpoint_dir, checkpoint_format, checkpoint_variant
            )
        try:
            return original_loader(
                model, checkpoint_dir, checkpoint_format, checkpoint_variant
            )
        except RuntimeError as exc:
            if "value_head." not in str(exc):
                raise
        # Re-load allowing only optional prefixes + value_head.* to be missing.
        native_weights = openpi_pkg._resolve_native_weights(checkpoint_dir)
        if native_weights is None:
            raise FileNotFoundError(
                f"No native OpenPI weights found under {checkpoint_dir}"
            )
        try:
            state_dict = torch.load(native_weights, map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(native_weights, map_location="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        allowed_prefixes = ("cosmos_prefix_projector.", "world_pilot_", "value_head.")
        missing_without_optional = sorted(
            key for key in missing if not key.startswith(allowed_prefixes)
        )
        if unexpected or missing_without_optional:
            raise RuntimeError(
                "Native OpenPI checkpoint is incompatible with the requested model: "
                f"missing={missing_without_optional[:20]}, "
                f"unexpected={list(unexpected)[:20]}"
            )
        if seed is not None and hasattr(model, "value_head") and any(
            key.startswith("value_head.") for key in missing
        ):
            # Deterministically initialize value_head across ranks before first FSDP/rollout sync.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(seed))
                for submodule in model.value_head.modules():
                    if hasattr(submodule, "reset_parameters"):
                        submodule.reset_parameters()

    _compat_load_state_dict._hybrid_value_head_compat = True
    openpi_pkg._load_state_dict = _compat_load_state_dict


def _is_value_head_param(name: str) -> bool:
    return "value_head" in name.split(".") or "value_head." in name


def snapshot_param_grads_to_cpu(
    model: torch.nn.Module,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Copy current parameter gradients to CPU, separating actor vs ``value_head``.

    Keeping the real-stream gradient snapshot on CPU adds zero GPU VRAM overhead
    while preserving exact FSDP-reduced gradients for conflict-free fusion.
    """
    actor_grads: dict[str, torch.Tensor] = {}
    critic_grads: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad_cpu = param.grad.detach().to(device="cpu", copy=True)
        if _is_value_head_param(name):
            critic_grads[name] = grad_cpu
        else:
            actor_grads[name] = grad_cpu
    return actor_grads, critic_grads


def merge_real_and_wm_gradients(
    model: torch.nn.Module,
    real_actor_grads_cpu: dict[str, torch.Tensor],
    real_critic_grads_cpu: dict[str, torch.Tensor],
    *,
    rho: float,
    grad_norm_match: bool = True,
    grad_conflict_mode: str = "cosine_gate",
    grad_cos_min: float = 0.0,
    is_sharded_fsdp: bool = False,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Fuse CPU-stashed real PPO gradients with current GPU WM-GRPO gradients in-place.

    Expects ``model`` parameters to currently hold the FSDP-reduced WM-GRPO
    gradients (with ``value_head`` grads absent because WM runs with
    ``compute_values=False``).

    Returns diagnostic scalars:
    - ``hybrid/grad_cos_sim``: cosine similarity ``cos(g_real, g_wm)``
    - ``hybrid/grad_real_norm``, ``hybrid/grad_wm_norm``, ``hybrid/grad_norm_ratio``
    - ``hybrid/wm_grad_accepted``, ``hybrid/wm_grad_projected``, ``hybrid/effective_wm_scale``
    """
    if grad_conflict_mode not in SUPPORTED_GRAD_CONFLICT_MODES:
        raise ValueError(f"Unsupported grad_conflict_mode: {grad_conflict_mode!r}")

    named_params = dict(model.named_parameters())

    real_sq = 0.0
    wm_sq = 0.0
    dot_val = 0.0
    actor_names = sorted(
        set(real_actor_grads_cpu.keys())
        | {
            name
            for name, param in named_params.items()
            if param.grad is not None and not _is_value_head_param(name)
        }
    )

    with torch.no_grad():
        for name in actor_names:
            param = named_params.get(name)
            if param is None:
                continue
            g_r_cpu = real_actor_grads_cpu.get(name)
            g_w = param.grad
            if g_r_cpu is not None:
                g_r_f = g_r_cpu.to(device=param.device, dtype=torch.float32)
                real_sq += float(torch.sum(g_r_f * g_r_f).item())
            else:
                g_r_f = None
            if g_w is not None:
                g_w_f = g_w.detach().to(dtype=torch.float32)
                wm_sq += float(torch.sum(g_w_f * g_w_f).item())
                if g_r_f is not None:
                    dot_val += float(torch.sum(g_r_f * g_w_f).item())

        if is_sharded_fsdp and torch.distributed.is_initialized():
            stats = torch.tensor(
                [real_sq, wm_sq, dot_val],
                dtype=torch.float64,
                device=next(model.parameters()).device,
            )
            torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
            real_sq, wm_sq, dot_val = stats.cpu().tolist()

        real_norm = math.sqrt(max(real_sq, 0.0))
        wm_norm = math.sqrt(max(wm_sq, 0.0))
        finite_ok = math.isfinite(real_norm) and math.isfinite(wm_norm) and math.isfinite(dot_val)
        if finite_ok and real_norm > eps and wm_norm > eps:
            cos_sim = max(-1.0, min(1.0, dot_val / (real_norm * wm_norm)))
            norm_ratio = wm_norm / real_norm
        else:
            cos_sim = 0.0
            norm_ratio = 0.0

        # Determine exact linear combination: g_total = alpha_real * g_real + alpha_wm * g_wm
        alpha_real = 1.0
        alpha_wm = 0.0
        accepted = 0.0
        projected = 0.0
        if finite_ok and rho > 0.0 and real_norm > eps and wm_norm > eps:
            scale = (real_norm / (wm_norm + eps)) if grad_norm_match else 1.0
            if grad_conflict_mode == "cosine_gate":
                if cos_sim >= grad_cos_min:
                    alpha_wm = float(rho) * scale
                    accepted = 1.0
            elif grad_conflict_mode == "pcgrad":
                accepted = 1.0
                if cos_sim < grad_cos_min:
                    # Remove conflicting component along g_real:
                    # g_wm_perp = s*g_wm - ((s*<g_wm,g_real>)/||g_real||^2)*g_real
                    #           = s*g_wm - (s * wm_norm / real_norm * cos_sim)*g_real
                    proj_coef = (scale * wm_norm / (real_norm + eps)) * cos_sim
                    alpha_real = 1.0 - float(rho) * proj_coef
                    alpha_wm = float(rho) * scale
                    projected = 1.0
                else:
                    alpha_wm = float(rho) * scale
            else:  # "none"
                alpha_wm = float(rho) * scale
                accepted = 1.0

        # Write fused actor gradients in-place.
        for name in actor_names:
            param = named_params.get(name)
            if param is None:
                continue
            g_r_cpu = real_actor_grads_cpu.get(name)
            g_w = param.grad
            if g_r_cpu is None and g_w is None:
                continue
            if alpha_wm == 0.0:
                if g_r_cpu is not None:
                    g_r_dev = g_r_cpu.to(device=param.device, dtype=param.dtype)
                    if param.grad is None:
                        param.grad = g_r_dev.clone()
                    else:
                        param.grad.copy_(g_r_dev)
                else:
                    param.grad = None
                continue

            fused = None
            if g_r_cpu is not None:
                fused = g_r_cpu.to(device=param.device, dtype=torch.float32) * alpha_real
            if g_w is not None:
                wm_term = g_w.detach().to(dtype=torch.float32) * alpha_wm
                fused = wm_term if fused is None else fused.add_(wm_term)
            if fused is not None:
                if param.grad is None:
                    param.grad = fused.to(dtype=param.dtype)
                else:
                    param.grad.copy_(fused.to(dtype=param.dtype))

        # Restore pure real-PPO value_head gradients verbatim.
        for name, param in named_params.items():
            if not _is_value_head_param(name):
                continue
            g_c_cpu = real_critic_grads_cpu.get(name)
            if g_c_cpu is None:
                param.grad = None
            else:
                g_c_dev = g_c_cpu.to(device=param.device, dtype=param.dtype)
                if param.grad is None:
                    param.grad = g_c_dev.clone()
                else:
                    param.grad.copy_(g_c_dev)

    return {
        "hybrid/grad_cos_sim": float(cos_sim),
        "hybrid/grad_real_norm": float(real_norm),
        "hybrid/grad_wm_norm": float(wm_norm),
        "hybrid/grad_norm_ratio": float(norm_ratio),
        "hybrid/wm_grad_accepted": float(accepted),
        "hybrid/wm_grad_projected": float(projected),
        "hybrid/effective_wm_scale": float(alpha_wm),
    }


class HybridSpatialEnvWorkerMixin:
    """Extend RLinf's ``EnvWorker`` with a real LIBERO training stream."""

    def _hybrid_enabled(self) -> bool:
        return bool(
            OmegaConf.select(
                self.cfg, "algorithm.hybrid_ppo_grpo.enabled", default=False
            )
        )

    def init_worker(self):
        super().init_worker()
        self.real_env_list = []
        if not self.enable_train or not self._hybrid_enabled():
            return
        from rlinf.envs import get_env_cls

        real_cfg = self.cfg.env.real_train
        self.real_rollout_epoch = int(real_cfg.rollout_epoch)
        self.real_num_envs_per_stage = (
            int(real_cfg.total_num_envs) // self._world_size // self.stage_num
        )
        self.real_batch_size = int(real_cfg.total_num_envs) // self.stage_num
        self.n_real_chunk_steps = (
            int(real_cfg.max_steps_per_rollout_epoch)
            // int(self.model_cfg.num_action_chunks)
        )
        self.real_prev_done = [
            torch.zeros(self.real_num_envs_per_stage, dtype=torch.bool)
            for _ in range(self.stage_num)
        ]
        real_env_cls = get_env_cls(real_cfg.env_type, real_cfg)
        self.real_env_list = self._setup_env_and_wrappers(
            env_cls=real_env_cls,
            env_cfg=real_cfg,
            num_envs_per_stage=self.real_num_envs_per_stage,
        )

    async def interact_real(
        self,
        input_channel,
        rollout_channel,
        reward_channel,
        actor_channel,
    ):
        """Run one real-LIBERO interaction pass for the PPO stream."""
        if not self._hybrid_enabled() or not self.real_env_list:
            raise RuntimeError("interact_real called when hybrid_ppo_grpo is disabled")

        from omegaconf import open_dict

        saved = (
            self.env_list,
            self.rollout_epoch,
            self.n_train_chunk_steps,
            self.train_num_envs_per_stage,
            self.train_batch_size,
            self.train_prev_done,
            self.train_enable_offload,
            self.cfg.env.train,
            getattr(self, "rollout_results", None),
            self._prefetched_train_bootstrap,
        )
        try:
            self.env_list = self.real_env_list
            self.rollout_epoch = self.real_rollout_epoch
            self.n_train_chunk_steps = self.n_real_chunk_steps
            self.train_num_envs_per_stage = self.real_num_envs_per_stage
            self.train_batch_size = self.real_batch_size
            self.train_prev_done = self.real_prev_done
            self.train_enable_offload = bool(
                self.cfg.env.real_train.get("enable_offload", False)
            )
            with open_dict(self.cfg.env):
                self.cfg.env.train = self.cfg.env.real_train
            self.rollout_results = None
            self._prefetched_train_bootstrap = None
            return await self.interact(
                input_channel=input_channel,
                rollout_channel=rollout_channel,
                reward_channel=reward_channel,
                actor_channel=actor_channel,
            )
        finally:
            (
                self.env_list,
                self.rollout_epoch,
                self.n_train_chunk_steps,
                self.train_num_envs_per_stage,
                self.train_batch_size,
                self.train_prev_done,
                self.train_enable_offload,
                prev_train_cfg,
                self.rollout_results,
                self._prefetched_train_bootstrap,
            ) = saved
            with open_dict(self.cfg.env):
                self.cfg.env.train = prev_train_cfg


class HybridRolloutMixin:
    """Extend ``CachedWorldModelRollout`` with a matching real-LIBERO generate pass."""

    async def generate_real(self, input_channel, output_channel):
        real_cfg = self.cfg.env.real_train
        saved = (
            self.rollout_epoch,
            self.n_train_chunk_steps,
            self.train_batch_size,
        )
        try:
            self.rollout_epoch = int(real_cfg.rollout_epoch)
            self.n_train_chunk_steps = (
                int(real_cfg.max_steps_per_rollout_epoch)
                // int(self.model_cfg.num_action_chunks)
            )
            self.train_batch_size = (
                int(real_cfg.total_num_envs) // int(self.num_pipeline_stages)
            )
            return await self.generate(
                input_channel=input_channel,
                output_channel=output_channel,
            )
        finally:
            (
                self.rollout_epoch,
                self.n_train_chunk_steps,
                self.train_batch_size,
            ) = saved


class HybridRunnerMixin:
    """Two-stream runner: Real-Env PPO (+ critic warmup) + Closed-Loop WM GRPO."""

    def _hybrid_enabled(self) -> bool:
        return bool(
            OmegaConf.select(
                self.cfg, "algorithm.hybrid_ppo_grpo.enabled", default=False
            )
        )

    def run(self):
        if not self._hybrid_enabled():
            return super().run()

        from rlinf.utils.metric_utils import compute_evaluate_metrics

        start_step = self.global_step
        start_time = time.time()
        for _step in range(start_step, self.max_steps):
            self.actor.set_global_step(self.global_step)
            self.rollout.set_global_step(self.global_step)

            with self.timer("step", trace_args={"step_idx": _step}):
                with self.timer("sync_weights"):
                    if self._should_sync_rollout_weights(_step):
                        self.update_rollout_weights()

                # Stream 1: Real LIBERO simulator rollout -> PPO (GAE + ValueHead).
                with self.timer("generate_real_rollouts"):
                    real_env_handle = self.env.interact_real(
                        input_channel=self.env_channel,
                        rollout_channel=self.rollout_channel,
                        reward_channel=self.reward_channel,
                        actor_channel=self.actor_channel,
                    )
                    real_rollout_handle = self.rollout.generate_real(
                        input_channel=self.rollout_channel,
                        output_channel=self.env_channel,
                    )
                    self.actor.recv_real_rollout_trajectories(
                        input_channel=self.actor_channel
                    ).wait()
                    real_rollout_handle.wait()
                    real_env_results = real_env_handle.wait()

                with self.timer("cal_real_adv_and_returns"):
                    real_rollout_metrics = (
                        self.actor.compute_real_advantages_and_returns().wait()
                    )

                run_wm_flags = self.actor.should_run_wm_stream().wait()
                run_wm = bool(run_wm_flags[0]) if isinstance(run_wm_flags, list) else bool(run_wm_flags)

                # Stream 2: Closed-loop Wan + Cosmos wrist rollout -> GRPO.
                wm_env_results = None
                if run_wm:
                    with self.timer("generate_wm_rollouts"):
                        wm_env_handle = self.env.interact(
                            input_channel=self.env_channel,
                            rollout_channel=self.rollout_channel,
                            reward_channel=self.reward_channel,
                            actor_channel=self.actor_channel,
                        )
                        wm_rollout_handle = self.rollout.generate(
                            input_channel=self.rollout_channel,
                            output_channel=self.env_channel,
                        )
                        self.actor.recv_rollout_trajectories(
                            input_channel=self.actor_channel
                        ).wait()
                        wm_rollout_handle.wait()
                        # Wait for Wan offload before advantages or actor CUDA updates.
                        wm_env_results = wm_env_handle.wait()
                        if hasattr(self, "_wait_for_update_memory"):
                            self._wait_for_update_memory()

                    with self.timer("cal_wm_adv_and_returns"):
                        wm_rollout_metrics = (
                            self.actor.compute_advantages_and_returns().wait()
                        )
                else:
                    wm_rollout_metrics = self.actor.skip_wm_stream().wait()

                # Joint Real-PPO + rho(H_t) * WM-GRPO update (or critic-only warmup).
                with self.timer("actor_training"):
                    actor_training_handle = self.actor.run_training()
                    actor_training_metrics = actor_training_handle.wait()

                self.global_step += 1
                eval_metrics = self._maybe_eval_and_checkpoint(_step)

            time_metrics = {
                f"time/{k}": v for k, v in self.timer.consume_durations().items()
            }
            real_env_list = [r for r in real_env_results if r is not None]
            real_env_metrics = (
                {f"env_real/{k}": v for k, v in compute_evaluate_metrics(real_env_list).items()}
                if real_env_list
                else {}
            )
            wm_env_metrics = {}
            if wm_env_results is not None:
                wm_env_list = [r for r in wm_env_results if r is not None]
                if wm_env_list:
                    wm_env_metrics = {
                        f"env/{k}": v
                        for k, v in compute_evaluate_metrics(wm_env_list).items()
                    }

            rollout_metrics = {
                f"rollout_real/{k}": v
                for k, v in self._aggregate_numeric_metrics(real_rollout_metrics).items()
            }
            rollout_metrics.update(
                {
                    f"rollout/{k}": v
                    for k, v in self._aggregate_numeric_metrics(wm_rollout_metrics).items()
                }
            )
            training_metrics = {
                f"train/{k}": v
                for k, v in self._aggregate_numeric_metrics(actor_training_metrics).items()
            }

            for payload in (
                real_env_metrics,
                wm_env_metrics,
                rollout_metrics,
                time_metrics,
                training_metrics,
            ):
                if payload:
                    self.metric_logger.log(payload, _step)

            logging_metrics = {}
            for payload in (
                time_metrics,
                eval_metrics,
                real_env_metrics,
                wm_env_metrics,
                rollout_metrics,
                training_metrics,
            ):
                logging_metrics.update(payload)
            self.print_metrics_table_async(
                _step, self.max_steps, start_time, logging_metrics, start_step
            )

        self._finish_run()
