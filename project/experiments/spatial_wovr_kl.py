"""Reference-policy KL regularization for embodied world-model GRPO actors."""

from __future__ import annotations

from contextlib import contextmanager
import math
from typing import Any

import torch
import torch.nn as nn

SUPPORTED_DIRECT_KL_PENALTIES = frozenset({"low_var_kl", "k3", "mse", "k2", "abs"})


def validate_kl_config(algorithm_cfg: Any) -> tuple[float, str]:
    """Return validated ``(kl_beta, kl_penalty_type)`` for direct KL loss."""
    raw_beta = algorithm_cfg.get("kl_beta", 0.0) if algorithm_cfg is not None else 0.0
    if isinstance(raw_beta, bool):
        raise ValueError("algorithm.kl_beta must be a finite nonnegative float")
    kl_beta = float(raw_beta)
    if not math.isfinite(kl_beta) or kl_beta < 0.0:
        raise ValueError("algorithm.kl_beta must be a finite nonnegative float")

    raw_penalty = None
    if algorithm_cfg is not None:
        raw_penalty = algorithm_cfg.get("kl_penalty_type", None)
        if raw_penalty is None:
            raw_penalty = algorithm_cfg.get("kl_penalty", "low_var_kl")
    kl_penalty_type = str(raw_penalty or "low_var_kl")

    if kl_beta > 0.0 and kl_penalty_type not in SUPPORTED_DIRECT_KL_PENALTIES:
        raise ValueError(
            "Direct actor KL loss requires kl_penalty in "
            f"{sorted(SUPPORTED_DIRECT_KL_PENALTIES)}, got {kl_penalty_type!r}. "
            "Linear 'kl'/'k1' (logprob - ref_logprob) has constant gradient +1 "
            "on fixed rollout samples and cannot be differentiated directly as a loss."
        )
    return kl_beta, kl_penalty_type


def snapshot_trainable_weights(
    model: nn.Module,
    *,
    include_names: list[str] | set[str] | tuple[str, ...] | None = None,
    exclude_substrings: tuple[str, ...] = (),
) -> dict[str, torch.Tensor]:
    """Copy trainable policy parameters to CPU as the fixed reference policy.

    When ``include_names`` is provided (e.g. ``store_requires_grad_param_name``
    while critic warmup has temporarily frozen actor parameters), those named
    parameters are captured even if ``requires_grad`` is currently ``False``.
    """
    include_set = set(include_names) if include_names else None
    ref_weights = {}
    for name, param in model.named_parameters():
        if exclude_substrings and any(sub in name for sub in exclude_substrings):
            continue
        if include_set is not None:
            if name not in include_set:
                continue
        elif not param.requires_grad:
            continue
        ref_weights[name] = param.detach().to(device="cpu", copy=True)
    if not ref_weights:
        raise ValueError("Cannot snapshot reference policy: model has no trainable parameters")
    return ref_weights


@contextmanager
def swap_trainable_weights(model: nn.Module, ref_weights: dict[str, torch.Tensor]):
    """Temporarily copy CPU reference weights into matching parameters in-place.

    In-place ``param.data.copy_`` preserves parameter identity, optimizer state
    bindings, and PyTorch FSDP ``FlatParameter`` storage/views.
    """
    named_params = dict(model.named_parameters())
    missing = [name for name in ref_weights if name not in named_params]
    if missing:
        raise ValueError(
            f"Reference policy parameters missing from model: {missing[:8]}"
        )
    target_params = {name: named_params[name] for name in ref_weights}
    for name, param in target_params.items():
        if tuple(param.shape) != tuple(ref_weights[name].shape):
            raise ValueError(
                f"Reference weight shape mismatch for {name}: "
                f"expected {tuple(param.shape)}, got {tuple(ref_weights[name].shape)}"
            )

    saved_current: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, param in target_params.items():
            saved_current[name] = param.detach().to(device="cpu", copy=True)
            param.data.copy_(
                ref_weights[name].to(device=param.device, dtype=param.dtype)
            )
    try:
        yield
    finally:
        with torch.no_grad():
            for name, param in target_params.items():
                if name in saved_current:
                    param.data.copy_(
                        saved_current[name].to(device=param.device, dtype=param.dtype)
                    )


def compute_kl_divergence(
    logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    kl_penalty_type: str,
) -> torch.Tensor:
    """Compute per-element KL penalty matching ``rlinf.algorithms.utils.kl_penalty``."""
    if kl_penalty_type in ("low_var_kl", "k3"):
        kl = torch.clamp(ref_logprobs - logprobs, min=-20.0, max=20.0)
        kld = (torch.exp(kl) - kl - 1.0).contiguous()
        return torch.clamp(kld, min=-10.0, max=10.0)
    if kl_penalty_type in ("mse", "k2"):
        return 0.5 * (logprobs - ref_logprobs).square()
    if kl_penalty_type == "abs":
        return (logprobs - ref_logprobs).abs()
    raise ValueError(f"Unsupported direct KL penalty type: {kl_penalty_type!r}")


def _expand_to_target_dim(
    tensor: torch.Tensor | None, target_shape: torch.Size
) -> torch.Tensor | None:
    if tensor is None:
        return None
    while tensor.ndim < len(target_shape):
        tensor = tensor.unsqueeze(-1)
    return tensor


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if (~mask).all():
        return (values * mask).sum()
    return (values * mask).sum() / mask.sum()


def compute_embodied_kl_loss(
    logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    *,
    kl_beta: float,
    kl_penalty_type: str,
    logprob_type: str,
    reward_type: str,
    single_action_dim: int,
    loss_mask: torch.Tensor | None = None,
    loss_mask_sum: torch.Tensor | None = None,
    max_episode_steps: int | None = None,
    critic_warmup: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute embodied actor KL penalty aligned with RLinf's policy-loss masking."""
    if reward_type == "chunk_level":
        if loss_mask is not None:
            loss_mask = loss_mask.flatten()
        if loss_mask_sum is not None:
            loss_mask_sum = loss_mask_sum.flatten()

    bsz = logprobs.shape[0]
    ref_logprobs = ref_logprobs.detach().to(device=logprobs.device, dtype=torch.float32)
    logprobs = logprobs.to(dtype=torch.float32)

    if logprob_type == "token_level":
        logprobs = logprobs.reshape(bsz, -1, single_action_dim)
        ref_logprobs = ref_logprobs.reshape(bsz, -1, single_action_dim)
        if loss_mask is not None:
            loss_mask = loss_mask.unsqueeze(-1)
        if loss_mask_sum is not None:
            loss_mask_sum = loss_mask_sum.unsqueeze(-1)
    elif logprob_type == "action_level":
        logprobs = logprobs.reshape(bsz, -1, single_action_dim).sum(dim=-1)
        ref_logprobs = ref_logprobs.reshape(bsz, -1, single_action_dim).sum(dim=-1)
    elif logprob_type == "chunk_level":
        logprobs = logprobs.reshape(bsz, -1, single_action_dim).sum(dim=[1, 2])
        ref_logprobs = ref_logprobs.reshape(bsz, -1, single_action_dim).sum(dim=[1, 2])
    else:
        raise ValueError(f"Unsupported logprob_type for KL loss: {logprob_type!r}")

    target_shape = logprobs.shape
    loss_mask = _expand_to_target_dim(loss_mask, target_shape)
    loss_mask_sum = _expand_to_target_dim(loss_mask_sum, target_shape)

    if loss_mask is None:
        loss_mask_bool = torch.ones_like(logprobs, dtype=torch.bool)
    else:
        loss_mask_bool = loss_mask.bool()

    kld = compute_kl_divergence(logprobs, ref_logprobs, kl_penalty_type)

    if max_episode_steps is not None and loss_mask_sum is not None:
        loss_mask_ratio = (loss_mask_sum.to(dtype=torch.float32) * 1.0) / float(
            max_episode_steps
        )
        kl_loss = (kld / loss_mask_ratio * loss_mask_bool).mean()
    else:
        kl_loss = _masked_mean(kld, loss_mask_bool.expand_as(kld))

    if critic_warmup or kl_beta <= 0.0:
        kl_penalty_loss = torch.zeros((), device=logprobs.device, dtype=logprobs.dtype)
    else:
        kl_penalty_loss = kl_loss * float(kl_beta)

    with torch.no_grad():
        ref_kl_mean = _masked_mean(kld.detach(), loss_mask_bool.expand_as(kld))

    metrics = {
        "actor/kl_loss": kl_loss.detach().item(),
        "actor/kl_penalty_loss": kl_penalty_loss.detach().item(),
        "actor/ref_kl_mean": ref_kl_mean.detach().item(),
    }
    return kl_penalty_loss, metrics


def _flatten_nested_time_batch(value: Any, key_path: tuple[str, ...] = ()) -> Any:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.dim() < 2:
            raise ValueError(
                f"Cannot flatten field {'.'.join(key_path)!r} with shape {tuple(value.shape)}"
            )
        return value.reshape(-1, *value.shape[2:])
    if isinstance(value, dict):
        return {
            key: _flatten_nested_time_batch(nested, (*key_path, key))
            for key, nested in value.items()
        }
    raise TypeError(
        f"Unsupported field {'.'.join(key_path)!r} type {type(value)} for KL scoring"
    )


def _split_nested_chunks(nested: dict[str, Any], num_chunks: int) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = [{} for _ in range(num_chunks)]
    for key, value in nested.items():
        if value is None:
            for chunk in chunks:
                chunk[key] = None
        elif isinstance(value, torch.Tensor):
            splits = torch.chunk(value, num_chunks, dim=0)
            for idx, part in enumerate(splits):
                chunks[idx][key] = part
        elif isinstance(value, dict):
            sub_chunks = _split_nested_chunks(value, num_chunks)
            for idx, part in enumerate(sub_chunks):
                chunks[idx][key] = part
        else:
            raise TypeError(f"Unsupported forward_inputs value for {key!r}: {type(value)}")
    return chunks


def _move_nested_to_device(nested: Any, device: Any) -> Any:
    if isinstance(nested, torch.Tensor):
        return nested.to(device)
    if isinstance(nested, dict):
        return {key: _move_nested_to_device(val, device) for key, val in nested.items()}
    return nested


def compute_ref_logprobs(
    model: nn.Module,
    rollout_batch: dict[str, Any],
    ref_weights: dict[str, torch.Tensor],
    *,
    device: Any,
    micro_batch_size: int,
    amp_context: Any,
    forward_kwargs: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Evaluate fixed reference-policy logprobs on ``rollout_batch['forward_inputs']``."""
    prev_logprobs = rollout_batch["prev_logprobs"]
    time_dim, batch_dim = prev_logprobs.shape[:2]
    flat_batch_size = time_dim * batch_dim
    flat_forward_inputs = _flatten_nested_time_batch(
        rollout_batch["forward_inputs"], ("forward_inputs",)
    )
    num_chunks = max(1, (flat_batch_size + micro_batch_size - 1) // micro_batch_size)
    kwargs = dict(forward_kwargs or {})

    ref_logprobs_parts: list[torch.Tensor] = []
    was_training = model.training
    with swap_trainable_weights(model, ref_weights):
        model.eval()
        try:
            with torch.no_grad():
                for micro_batch in _split_nested_chunks(flat_forward_inputs, num_chunks):
                    micro_batch = _move_nested_to_device(micro_batch, device)
                    with amp_context:
                        ref_output = model(
                            forward_inputs=micro_batch,
                            compute_logprobs=True,
                            compute_entropy=False,
                            compute_values=False,
                            use_cache=False,
                            **kwargs,
                        )
                    ref_logprobs_parts.append(
                        ref_output["logprobs"].detach().to(dtype=torch.float32, device="cpu")
                    )
        finally:
            model.train(was_training)

    ref_logprobs = torch.cat(ref_logprobs_parts, dim=0)
    expected_shape = (flat_batch_size, *prev_logprobs.shape[2:])
    if tuple(ref_logprobs.shape) != expected_shape:
        raise ValueError(
            f"ref_logprobs shape {tuple(ref_logprobs.shape)} does not match "
            f"expected {expected_shape}"
        )
    return ref_logprobs.reshape(time_dim, batch_dim, *ref_logprobs.shape[1:])
