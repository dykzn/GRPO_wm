"""Skip inactive Cosmos denoising rows without changing native batch noise.

The native full-batch generator and solver run unchanged. Only the expensive
denoiser receives compact inputs, with its predictions scattered back before
the solver sees them. Conditional VAE encoding deliberately remains full batch.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import textwrap
from contextlib import contextmanager
from dataclasses import fields, is_dataclass, replace
from functools import lru_cache
from types import MethodType

import numpy as np
import torch

_MODEL_MODULE = "cosmos_policy.models.temporal_action_conditioned_model"
_SAMPLER_MODULE = "cosmos_policy._src.imaginaire.modules.res_sampler"
# Canonical AST hashes pin the inspected production protocol. Whitespace and
# line numbers do not matter; an altered generator, solver or denoiser does.
_SOURCE_HASHES = {
    "denoise": "27d99d41f04194770439bab19b1eb641a68f9bb850fc337af4bdced18e0cc317",
    "generate_samples_from_batch": "ed03473ee5df865a6a309e4a822ef3f796a875ad0bbdf766db1a1ea8d8bf3a6d",
    "get_x0_fn_from_batch": "6b3d0c9254cb7433a422e768761592e68d9d2d9affbbf055b6e3ae9a81633724",
    "forward": "db83359c46bb8f11d25a178d986c12723c9ef9288a3d15af173bb3af2aadbdc7",
    "_forward_impl": "7fe3cebd3e4c017c4c33e1fc2b81a75d9c33649089524770e7192bd2c76b30ef",
}
_BATCH_CONDITION_FIELDS = frozenset({
    "crossattn_emb", "padding_mask", "fps", "gt_frames",
    "condition_video_input_mask_B_C_T_H_W", "action", "img_context_emb",
})
_SCALAR_CONDITION_FIELDS = frozenset({
    "_is_broadcasted", "data_type", "use_video_condition",
})


def validate_active_sample_mask(value, batch_size: int) -> np.ndarray:
    """Reject coercion of numeric/string masks; keep original sample positions."""
    mask = np.asarray(value)
    if mask.dtype != np.bool_ or mask.shape != (batch_size,) or batch_size < 1:
        raise ValueError(
            "active_sample_mask must be bool[B] with "
            f"B={batch_size}, got dtype={mask.dtype}, shape={mask.shape}"
        )
    return np.ascontiguousarray(mask)


@lru_cache(maxsize=32)
def _check_method_source(function, name: str) -> None:
    source = textwrap.dedent(inspect.getsource(function))
    tree = ast.parse(source)
    digest = hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()
    if digest != _SOURCE_HASHES[name]:
        raise RuntimeError(f"Unsupported Cosmos {name} implementation for active-row skipping")


def _validate_cosmos_protocol(model) -> None:
    if (type(model).__module__, type(model).__name__) != (
        _MODEL_MODULE, "CosmosWoVRVideo2WorldModel"
    ):
        raise RuntimeError("Active-row skipping requires the inspected Cosmos WoVR wrist model")
    config, net = model.config, model.net
    net = getattr(net, "module", net)
    if (
        model.training
        or getattr(net, "training", False)
        or config.state_t != 4
        or config.min_num_conditional_frames != 2
        or config.use_flowunipc_scheduler
        or net.is_context_parallel_enabled
        or not net.wovr_action_contract
        or getattr(net, "cross_view_primary_context", False)
    ):
        raise RuntimeError("Unsupported Cosmos configuration for active-row skipping")
    for name in ("generate_samples_from_batch", "get_x0_fn_from_batch", "denoise"):
        function = getattr(getattr(model, name), "__func__", None)
        if function is None:
            raise RuntimeError(f"Unsupported Cosmos bound method: {name}")
        _check_method_source(function, name)
    sampler = model.sampler
    if (type(sampler).__module__, type(sampler).__name__) != (_SAMPLER_MODULE, "Sampler"):
        raise RuntimeError("Active-row skipping requires the inspected native Cosmos sampler")
    for name in ("forward", "_forward_impl"):
        function = getattr(getattr(sampler, name), "__func__", None)
        if function is None:
            raise RuntimeError(f"Unsupported Cosmos sampler method: {name}")
        _check_method_source(function, name)


def cosmos_active_sample_mask_supported(model) -> bool:
    """Advertise support only when the loaded instance matches the protocol."""
    try:
        _validate_cosmos_protocol(model)
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return False
    return True


@contextmanager
def compact_cosmos_denoiser(model, active_sample_mask):
    """Compact a single model's denoiser for exactly one synchronous request.

    Native full-batch initial noise, solver noise, variance scaling, conditions
    and seeds stay in their original positions. Inactive predictions are zero
    placeholders and must be discarded by the caller. No global RNG or class
    method is changed, and the original instance attribute is restored even
    when conditioning, generation or denoising raises.
    """
    mask = validate_active_sample_mask(active_sample_mask, len(active_sample_mask))
    if not mask.any():
        raise ValueError("The compact denoiser requires at least one active sample")
    if getattr(model, "_spatial_wovr_compact_request", False):
        raise RuntimeError("Overlapping compact Cosmos requests are unsupported")
    if mask.all():
        yield
        return
    _validate_cosmos_protocol(model)
    batch_size = len(mask)
    original = model.denoise
    absent = object()
    previous_attribute = model.__dict__.get("denoise", absent)
    indices_by_device = {}
    condition_cache = {}

    def indices(device):
        if device not in indices_by_device:
            indices_by_device[device] = torch.as_tensor(
                np.flatnonzero(mask), dtype=torch.int64, device=device
            )
        return indices_by_device[device]

    def select_batch(value, name):
        if value is None:
            return None
        if not isinstance(value, torch.Tensor) or not value.ndim or value.shape[0] != batch_size:
            raise RuntimeError(f"Unsupported Cosmos batch field {name}")
        return value.index_select(0, indices(value.device))

    def select_condition(condition):
        cached = condition_cache.get(id(condition))
        if cached is not None and cached[0] is condition:
            return cached[1]
        expected = _BATCH_CONDITION_FIELDS | _SCALAR_CONDITION_FIELDS
        if not is_dataclass(condition) or {field.name for field in fields(condition)} != expected:
            raise RuntimeError("Unsupported Cosmos condition layout for active-row skipping")
        selected = replace(condition, **{
            name: select_batch(getattr(condition, name), name)
            for name in _BATCH_CONDITION_FIELDS
        })
        condition_cache[id(condition)] = (condition, selected)
        return selected

    def scatter(value, name):
        if value is None:
            return None
        if not isinstance(value, torch.Tensor) or not value.ndim or value.shape[0] != int(mask.sum()):
            raise RuntimeError(f"Unsupported Cosmos denoiser prediction {name}")
        result = value.new_zeros((batch_size, *value.shape[1:]))
        return result.index_copy_(0, indices(value.device), value)

    def denoise(_model, xt_B_C_T_H_W, sigma, condition):
        if xt_B_C_T_H_W.shape[0] != batch_size:
            raise RuntimeError("Cosmos solver changed the original batch size")
        if sigma.ndim not in (1, 2) or sigma.shape[0] not in (1, batch_size):
            raise RuntimeError("Unsupported Cosmos sigma layout for active-row skipping")
        selected_sigma = sigma if sigma.shape[0] == 1 else sigma.index_select(0, indices(sigma.device))
        prediction = original(
            xt_B_C_T_H_W.index_select(0, indices(xt_B_C_T_H_W.device)),
            selected_sigma,
            select_condition(condition),
        )
        if not is_dataclass(prediction) or {field.name for field in fields(prediction)} != {"x0", "eps", "logvar"}:
            raise RuntimeError("Unsupported Cosmos denoiser prediction layout")
        return replace(prediction, **{
            name: scatter(getattr(prediction, name), name)
            for name in ("x0", "eps", "logvar")
        })

    model._spatial_wovr_compact_request = True
    model.denoise = MethodType(denoise, model)
    try:
        yield
    finally:
        if previous_attribute is absent:
            del model.denoise
        else:
            model.denoise = previous_attribute
        del model._spatial_wovr_compact_request
