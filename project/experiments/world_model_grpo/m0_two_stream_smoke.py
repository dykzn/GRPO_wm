#!/usr/bin/env python3
"""M0 two-stream smoke: one real trajectory + 8 NFT counterfactual variants.

Data contract under test (one serial update step):

  1. real stream : 1 LIBERO rollout, simulator ground-truth success reward,
                   per-trajectory normalized advantage -> reward-weighted NFT loss;
  2. synthetic   : same trajectory's saved nft_x0, 8x resample
                   ``xcur = (1-t)*x0 + t*eps`` -> re-denoise with current
                   policy -> 8 open-loop counterfactual action sequences ->
                   Wan video from the same real o_0 -> goal-ResNet reward ->
                   group-normalized advantage -> same NFT loss;
  3. total       : ``L = L_NFT(real) + rho * L_NFT(syn)``, one optimizer step.

This script deliberately reuses the standalone policy/env/Wan machinery from
``experiments/audit_rlinf_pi05_wan_libero_pair.py`` instead of the RLinf
runner, so the M0 contract can be validated in isolation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import random
import sys
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = pathlib.Path(
    os.environ.get("PROJECT_ROOT", str(pathlib.Path(__file__).resolve().parents[2]))
).expanduser().resolve()
RLINF_ROOT = pathlib.Path(
    os.environ.get("RLINF_ROOT", str(PROJECT_ROOT / "third_party/RLinf"))
).expanduser().resolve()
WAN_ROOT = pathlib.Path(
    os.environ.get("WAN_ROOT", str(PROJECT_ROOT / "external/wan"))
).expanduser().resolve()
for _p in (
    PROJECT_ROOT,
    RLINF_ROOT,
    WAN_ROOT,
):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from experiments.audit_rlinf_pi05_wan_libero_pair import (  # noqa: E402
    CONDITION_FRAMES,
    FUTURE_FRAMES,
    ACTION_DIM,
    _ensure_openpi_tqdm_compat,
    _ensure_rlinf_ray_compat,
    labelled,
    make_env,
    output_to_uint8,
    policy_observation,
    rot180,
    save_video,
    settle_env,
)

from omegaconf import OmegaConf  # noqa: E402


def _np_image_to_pil(frame: np.ndarray) -> Any:
    from PIL import Image

    return Image.fromarray(frame.astype(np.uint8))


def build_model_context(model: Any, env_obs: dict[str, Any]) -> dict[str, Any]:
    """Replicate predict_action_batch's preprocessing up to the prefix cache."""
    from openpi.models import model as _model

    to_process_obs = model.obs_processor(env_obs)
    processed_obs = model.input_transform(to_process_obs, transpose=False)
    processed_obs = model.precision_processor(processed_obs)
    observation = _model.Observation.from_dict(processed_obs)
    images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(
        observation, train=False
    )
    prefix_output, prefix_pad_masks, past_key_values = model._build_prefix_cache(
        images, img_masks, lang_tokens, lang_masks
    )
    return {
        "observation": observation,
        "state": state,
        "prefix_pad_masks": prefix_pad_masks,
        "past_key_values": past_key_values,
    }


def denoise_from(
    model: Any,
    ctx: dict[str, Any],
    xcur: torch.Tensor,
    start_idx: int,
    num_steps: int,
    noise_method: str = "flow_ode",
    single_noise_step: bool | None = None,
) -> torch.Tensor:
    """Run the remaining denoising steps from ``xcur``.

    ``xcur`` is sampled immediately before ``start_idx``.  OpenPI's regular
    PPO rollout protocol injects noise at exactly one denoising step and uses
    ODE updates at every other step.  NFT rollouts intentionally use the
    configured method at every step.  If the caller does not specify the
    protocol, infer it from ``model.config.is_nft``.
    """
    if not 0 <= int(start_idx) < int(num_steps):
        raise ValueError(
            f"start_idx must be in [0, {int(num_steps)}), got {start_idx}"
        )
    if single_noise_step is None:
        single_noise_step = not bool(getattr(model.config, "is_nft", False))
    x_t = xcur
    for idx in range(start_idx, num_steps):
        sample_method = (
            noise_method
            if not single_noise_step or idx == start_idx
            else "flow_ode"
        )
        x_t_mean, x_t_std, _value, _v = model.sample_mean_var_val(
            x_t,
            idx,
            ctx["state"],
            ctx["prefix_pad_masks"],
            ctx["past_key_values"],
            sample_method,
            num_steps,
            False,
        )
        if sample_method == "flow_ode":
            x_t = x_t_mean
        else:
            x_t = x_t_mean + torch.randn_like(x_t_mean) * x_t_std
    return x_t


def velocity_at(
    model: Any, ctx: dict[str, Any], xcur: torch.Tensor, t: float
) -> torch.Tensor:
    t_tensor = torch.as_tensor([t], device=xcur.device, dtype=torch.float32)
    v, _suffix = model.get_velocity(
        ctx["state"], xcur, t_tensor, ctx["prefix_pad_masks"], ctx["past_key_values"]
    )
    # NFT is defined in the policy's full action space (Pi05 uses 32 dims).
    # The 7-D LIBERO action projection happens only in output_transform before
    # actions are sent to the simulator or Wan.
    return v[:, :FUTURE_FRAMES, :]


def nft_loss_for_stream(
    model: Any,
    frozen: Any,
    contexts: list[dict[str, Any]],
    xcur_list: list[torch.Tensor],
    t_list: list[float],
    x0_targets: list[torch.Tensor],
    advantages: list[float],
    *,
    form: str,
    max_drift: float,
    beta: float,
    dpo_beta: float,
    v_old_list: list[torch.Tensor] | None = None,
    noise_level_list: list[float] | None = None,
    step_index_list: list[int] | None = None,
    num_steps: int | None = None,
    nft_weight_mode: str = "constant",
    nft_weight_scale: float | tuple[float, float] = 1.0,
    nft_sum_type: str = "action_level",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Reward-weighted NFT loss over decision steps.

    With ``noise_level_list`` and ``step_index_list`` supplied, use the same
    SDE-aware x0 target/prediction correction and energy weighting as RLinf's
    NFT worker.  Omitting them preserves the original M0 ODE diagnostic path.
    """
    losses: list[torch.Tensor] = []
    if v_old_list is not None and len(v_old_list) != len(contexts):
        raise ValueError(
            "v_old_list must have one rollout velocity per NFT transition"
        )
    sde_nft = noise_level_list is not None or step_index_list is not None
    if nft_sum_type == "action_level":
        sum_dims = (2,)
    elif nft_sum_type == "chunk_level":
        sum_dims = (1, 2)
    else:
        raise ValueError(f"Unknown NFT sum type: {nft_sum_type}")
    if isinstance(nft_weight_scale, (tuple, list)):
        if len(nft_weight_scale) != 2:
            raise ValueError(
                "nft_weight_scale must be a scalar or [positive, negative]"
            )
        positive_scale, negative_scale = (
            float(nft_weight_scale[0]),
            float(nft_weight_scale[1]),
        )
    else:
        positive_scale = negative_scale = float(nft_weight_scale)
    if (
        not math.isfinite(positive_scale)
        or not math.isfinite(negative_scale)
        or positive_scale <= 0.0
        or negative_scale <= 0.0
    ):
        raise ValueError("NFT weight scales must be finite and positive")
    if sde_nft:
        if noise_level_list is None or step_index_list is None:
            raise ValueError(
                "noise_level_list and step_index_list must be supplied together"
            )
        if num_steps is None or num_steps <= 0:
            raise ValueError("num_steps must be positive for SDE-aware NFT")
        if len(noise_level_list) != len(contexts) or len(step_index_list) != len(contexts):
            raise ValueError(
                "SDE NFT metadata must have one noise level and step index "
                "per transition"
            )
    for index, (ctx, xcur, t, x0_tgt, adv) in enumerate(
        zip(contexts, xcur_list, t_list, x0_targets, advantages)
    ):
        v_theta = velocity_at(model, ctx, xcur, t)
        if v_old_list is None:
            with torch.no_grad():
                v_old = velocity_at(frozen, ctx, xcur, t)
        else:
            v_old = v_old_list[index].to(device=xcur.device, dtype=xcur.dtype)
        v_old = v_old[:, :FUTURE_FRAMES, :].detach()
        delta = v_theta - v_old
        norm = delta.norm(dim=sum_dims, keepdim=True) + 1e-8
        coef = (max_drift / norm).clamp(max=1.0)
        delta_c = delta * coef
        v_pos = v_old + beta * delta_c
        v_neg = v_old - beta * delta_c
        # Match RLinf's NFT worker: the model predicts the full action
        # horizon, but NFT loss is evaluated on the first action_chunk only.
        xcur_chunk = xcur[:, :FUTURE_FRAMES, :]
        x0_chunk = x0_tgt[:, :FUTURE_FRAMES, :]
        if sde_nft:
            schedule = torch.linspace(
                1.0,
                0.0,
                int(num_steps) + 1,
                device=xcur.device,
                dtype=xcur.dtype,
            )
            step_index = int(step_index_list[index])
            if not 0 <= step_index < int(num_steps):
                raise ValueError(
                    f"NFT step index {step_index} outside [0, {int(num_steps)})"
                )
            t_schedule = schedule[step_index]
            if abs(float(t_schedule.item()) - float(t)) > 1e-5:
                raise ValueError(
                    "NFT timestep metadata disagrees with num_steps schedule: "
                    f"t_list={t}, schedule={float(t_schedule.item())}"
                )
            dt = schedule[step_index] - schedule[step_index + 1]
            safe_schedule = schedule.clone()
            safe_schedule[0] = safe_schedule[1]
            sigma = torch.sqrt(
                schedule[:-1][step_index] / (1.0 - safe_schedule[:-1][step_index])
            ) * float(noise_level_list[index])
            t_bc = t_schedule.reshape(1, 1, 1)
            dt_bc = dt.reshape(1, 1, 1)
            sigma_bc = sigma.reshape(1, 1, 1)
            x0_pred_pos = xcur_chunk - t_bc * v_pos
            x0_pred_neg = xcur_chunk - t_bc * v_neg
            x1_pred_pos = xcur_chunk + (1.0 - t_bc) * v_pos
            x1_pred_neg = xcur_chunk + (1.0 - t_bc) * v_neg
            correction = (sigma_bc**2) / 2.0
            pred_pos = x0_pred_pos - correction * x1_pred_pos
            pred_neg = x0_pred_neg - correction * x1_pred_neg
            if nft_weight_mode == "auto":
                weight_mode = (
                    "adaptive"
                    if float(noise_level_list[index]) == 0.0
                    else "sigma"
                )
            else:
                weight_mode = nft_weight_mode
            std_t_det = torch.sqrt(dt_bc.clamp_min(0.0)) * sigma_bc
            if weight_mode == "constant":
                weight_pos = torch.as_tensor(
                    positive_scale, device=xcur.device, dtype=xcur.dtype
                )
                weight_neg = torch.as_tensor(
                    negative_scale, device=xcur.device, dtype=xcur.dtype
                )
            elif weight_mode == "t":
                weight_pos = positive_scale * t_bc**2
                weight_neg = negative_scale * t_bc**2
            elif weight_mode == "1-t":
                weight_pos = positive_scale * (1.0 - t_bc) ** 2
                weight_neg = negative_scale * (1.0 - t_bc) ** 2
            elif weight_mode == "inv_t2":
                weight_pos = positive_scale / (t_bc**2 + 1e-4)
                weight_neg = negative_scale / (t_bc**2 + 1e-4)
            elif weight_mode == "inv_t3":
                weight_pos = positive_scale / (t_bc**3 + 1e-4)
                weight_neg = negative_scale / (t_bc**3 + 1e-4)
            elif weight_mode == "sigma":
                weight_pos = positive_scale / (std_t_det**2 + 1e-4)
                weight_neg = negative_scale / (std_t_det**2 + 1e-4)
            elif weight_mode == "adaptive":
                with torch.no_grad():
                    weight_pos = positive_scale / (
                        (pred_pos - x0_chunk).double().abs().mean().clamp_min(1e-4)
                    ).to(xcur.dtype)
                    weight_neg = negative_scale / (
                        (pred_neg - x0_chunk).double().abs().mean().clamp_min(1e-4)
                    ).to(xcur.dtype)
            else:
                raise ValueError(f"Unknown NFT weight mode: {nft_weight_mode}")
            e_pos = ((pred_pos - x0_chunk) ** 2 * weight_pos).sum(dim=sum_dims)
            e_neg = ((pred_neg - x0_chunk) ** 2 * weight_neg).sum(dim=sum_dims)
        else:
            pred_pos = xcur_chunk - t * v_pos
            pred_neg = xcur_chunk - t * v_neg
            e_pos = ((pred_pos - x0_chunk) ** 2).sum(dim=sum_dims)
            e_neg = ((pred_neg - x0_chunk) ** 2).sum(dim=sum_dims)
        if form == "dpo":
            y = 2.0 * adv - 1.0
            logit = (dpo_beta / 2.0) * y * (e_pos - e_neg)
            losses.append(torch.nn.functional.softplus(logit).mean())
        elif form == "mse":
            losses.append((adv * e_pos + (1.0 - adv) * e_neg).mean())
        else:
            raise ValueError(f"Unknown loss form: {form}")
    loss = torch.stack(losses).mean()
    return loss, {}


def normalize_trajectory_advantages(rewards: list[float]) -> list[float]:
    r = np.asarray(rewards, dtype=np.float64)
    if r.max() - r.min() < 1e-12:
        return [0.5] * len(r)
    return ((r - r.min()) / (r.max() - r.min())).tolist()


def normalize_group_advantages(rewards: list[float], clip: float = 1.0) -> list[float]:
    r = np.asarray(rewards, dtype=np.float64)
    std = r.std()
    if std < 1e-12:
        return [0.5] * len(r)
    adv = np.clip((r - r.mean()) / std, -clip, clip)
    return ((adv + clip) / (2.0 * clip)).tolist()


def score_frames(rm: Any, frames: list[np.ndarray], instruction: str) -> float:
    """Score all generated frames and return the trajectory-level maximum.

    The native Wan environment evaluates every generated frame and uses an
    ``any-frame`` success rule.  Scoring only ``arr[-1:]`` made the standalone
    path disagree with that contract and could erase a success that occurred
    before the final frame.
    """
    if not frames:
        raise ValueError("cannot score an empty generated trajectory")
    arr = np.stack(frames).astype(np.float32) / 255.0 * 2.0 - 1.0
    arr = arr.transpose(0, 3, 1, 2)
    try:
        device = next(rm.parameters()).device
    except (AttributeError, StopIteration):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.from_numpy(arr).clamp(-1.0, 1.0).to(device, torch.float32)
    with torch.no_grad():
        values = rm.predict_rew(x, [instruction] * len(frames))
        values = torch.as_tensor(values).detach().float().reshape(-1)
        if values.numel() != len(frames):
            raise ValueError(
                "reward model returned one value per frame requirement violation: "
                f"got {values.numel()} values for {len(frames)} frames"
            )
        value = float(values.max().cpu().item())
    if not math.isfinite(value):
        raise ValueError(f"Non-finite reward: {value}")
    return value


def enforce_signed_gripper(actions: np.ndarray) -> np.ndarray:
    """Apply the LIBERO/Pi05 signed gripper boundary.

    This checkpoint's action statistics use ``-1=open`` and ``+1=closed``.
    Keep the six Cartesian dimensions unchanged and only clamp the final
    gripper command at the environment/Wan boundary.
    """
    values = np.asarray(actions, dtype=np.float32).copy()
    if values.shape[-1] != ACTION_DIM:
        raise ValueError(f"Expected [{ACTION_DIM}] LIBERO actions, got {values.shape}")
    values[..., -1] = np.clip(values[..., -1], -1.0, 1.0)
    return values


def save_libero_video(
    path: pathlib.Path, frames: list[np.ndarray], title: str
) -> None:
    """Save every observed LIBERO frame from the real rollout."""
    import imageio.v2 as imageio

    rendered = [
        labelled(frame, f"{title} h{index}")
        for index, frame in enumerate(frames, start=1)
    ]
    imageio.mimwrite(
        path,
        rendered,
        fps=2,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    )


def score_wan_variants(
    pipe: Any,
    reward_model: Any,
    reference_frame: np.ndarray,
    instruction: str,
    variants: list[dict[str, Any]],
    *,
    seed: int,
    num_inference_steps: int,
    output_dir: pathlib.Path | None = None,
    output_prefix: str = "wan_variant",
) -> tuple[list[float], list[list[np.ndarray]]]:
    """Generate and score every chunk of every counterfactual trajectory.

    Each branch keeps its own rolling image memory and previous-action
    condition.  A single final score is then assigned to the complete branch,
    so the synthetic reward has the same temporal extent as the NFT
    transitions it supervises.
    """
    if not variants:
        raise ValueError("at least one Wan variant is required")
    num_chunks = len(variants[0].get("chunks", ()))
    if num_chunks <= 0:
        raise ValueError("Wan variants must contain at least one action chunk")
    for variant_index, variant in enumerate(variants):
        chunks = variant.get("chunks", ())
        if len(chunks) != num_chunks:
            raise ValueError(
                "Wan variants must have the same number of chunks: "
                f"variant 0={num_chunks}, variant {variant_index}={len(chunks)}"
            )
        for chunk_index, chunk in enumerate(chunks):
            chunk_array = np.asarray(chunk, dtype=np.float32)
            expected_shape = (FUTURE_FRAMES, ACTION_DIM)
            if chunk_array.shape != expected_shape:
                raise ValueError(
                    f"variant {variant_index} chunk {chunk_index} has shape "
                    f"{chunk_array.shape}, expected {expected_shape}"
                )
            if not np.isfinite(chunk_array).all():
                raise ValueError(
                    f"variant {variant_index} chunk {chunk_index} contains non-finite values"
                )

    reference_frame = np.ascontiguousarray(reference_frame, dtype=np.uint8)
    sentinel = np.zeros((CONDITION_FRAMES, ACTION_DIM), dtype=np.float32)
    sentinel[:, -1] = -1.0
    num_variants = len(variants)
    memories = [np.repeat(reference_frame[None], 4, axis=0) for _ in variants]
    condition_actions = np.repeat(sentinel[None], num_variants, axis=0)
    frames_by_variant: list[list[np.ndarray]] = [
        [] for _ in variants
    ]

    for chunk_index in range(num_chunks):
        current_action_chunks = np.stack(
            [np.asarray(variant["chunks"][chunk_index], dtype=np.float32) for variant in variants],
            axis=0,
        )
        batch_actions = np.stack(
            [
                np.concatenate(
                    [condition_actions[variant_index], current_action_chunks[variant_index]],
                    axis=0,
                )
                for variant_index in range(num_variants)
            ],
            axis=0,
        )
        with torch.inference_mode(), torch.amp.autocast(
            "cuda", dtype=torch.bfloat16
        ):
            generated = pipe(
                seed=int(seed + 1000 * chunk_index),
                tiled=False,
                input_image=[_np_image_to_pil(reference_frame)] * num_variants,
                input_image4=[
                    [_np_image_to_pil(frame) for frame in memory]
                    for memory in memories
                ],
                action=torch.from_numpy(batch_actions).to("cuda", torch.float32),
                height=256,
                width=256,
                num_frames=CONDITION_FRAMES + FUTURE_FRAMES,
                num_inference_steps=num_inference_steps,
                cfg_scale=1.0,
                progress_bar_cmd=lambda values: values,
                batch_size=num_variants,
            )
        generated_batch = output_to_uint8(generated)[:, CONDITION_FRAMES:]
        del generated
        if generated_batch.shape[0] != num_variants:
            raise RuntimeError(
                "Wan returned a different batch size than requested: "
                f"{generated_batch.shape[0]} vs {num_variants}"
            )
        for variant_index in range(num_variants):
            generated_chunk = [
                np.ascontiguousarray(frame)
                for frame in generated_batch[variant_index]
            ]
            if len(generated_chunk) != FUTURE_FRAMES:
                raise RuntimeError(
                    "Wan returned a different chunk horizon: "
                    f"{len(generated_chunk)} vs {FUTURE_FRAMES}"
                )
            frames_by_variant[variant_index].extend(generated_chunk)
            memories[variant_index] = np.stack(generated_chunk[-4:], axis=0)
            condition_actions[variant_index, 1:, :] = current_action_chunks[
                variant_index, -4:
            ]

    rewards = [
        float(score_frames(reward_model, frames, instruction))
        for frames in frames_by_variant
    ]
    if output_dir is not None:
        for variant_index, frames in enumerate(frames_by_variant):
            save_video(
                output_dir / f"{output_prefix}_{variant_index:02d}_full.mp4",
                np.repeat(reference_frame[None], len(frames), axis=0),
                np.stack(frames, axis=0),
                f"Wan full synthetic variant {variant_index}",
            )
    return rewards, frames_by_variant


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--pi05-path", type=pathlib.Path, required=True)
    parser.add_argument(
        "--norm-stats-path",
        type=pathlib.Path,
        default=pathlib.Path(
            os.environ.get("CHECKPOINT_ROOT", str(PROJECT_ROOT / "checkpoints"))
        )
        / "pi05/RLinf-Pi05-LIBERO-SFT/physical-intelligence/libero",
        help="External OpenPI norm-stats directory for FSDP GRPO checkpoints.",
    )
    parser.add_argument("--wan-model-path", type=pathlib.Path, required=True)
    parser.add_argument("--vae-path", type=pathlib.Path, required=True)
    parser.add_argument("--goal-reward-path", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--task-suite", default="libero_goal")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--trial-id", type=int, default=0)
    parser.add_argument("--settle-steps", type=int, default=15)
    parser.add_argument("--max-decisions", type=int, default=4)
    parser.add_argument("--num-variants", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument(
        "--noise-method",
        choices=("flow_sde", "flow_ode"),
        default="flow_sde",
        help="Pi05 sampling protocol; the selected GRPO checkpoint was evaluated with flow_sde.",
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        default=0.3,
        help="Pi05 sampling noise level used by the GRPO evaluation protocol.",
    )
    parser.add_argument("--num-inference-steps", type=int, default=5)
    parser.add_argument("--loss-form", choices=("mse", "dpo"), default="mse")
    parser.add_argument("--rho", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--max-drift", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--dpo-beta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    cfg = OmegaConf.load(args.config)
    cfg.rollout.model.model_path = str(args.pi05_path)
    OmegaConf.update(
        cfg,
        "rollout.model.openpi_data.norm_stats_path",
        str(args.norm_stats_path),
        force_add=True,
    )
    cfg.rollout.model.num_action_chunks = FUTURE_FRAMES
    cfg.rollout.model.openpi.action_chunk = FUTURE_FRAMES
    cfg.rollout.model.openpi.num_steps = args.num_steps
    cfg.rollout.model.openpi.is_nft = True
    cfg.rollout.model.openpi.noise_method = args.noise_method
    cfg.rollout.model.openpi.noise_level = args.noise_level
    # RLinf-Pi05-LIBERO-SFT uses the signed action statistics directly:
    # -1=open and +1=closed. Do not apply the dataset [0,1] -> signed
    # conversion a second time at the OpenPI output boundary.
    OmegaConf.update(
        cfg,
        "rollout.model.openpi_data.output_gripper_action_mode",
        "already_signed",
        force_add=True,
    )

    _ensure_rlinf_ray_compat()
    _ensure_openpi_tqdm_compat()

    from rlinf.models.embodiment.openpi import get_model

    model = get_model(cfg.rollout.model).to("cuda").eval()
    frozen = get_model(cfg.rollout.model).to("cuda").eval()
    for param in frozen.parameters():
        param.requires_grad_(False)

    # ---------------- real stream ----------------
    from libero.libero import benchmark

    suite_cls = benchmark.get_benchmark_dict()[args.task_suite]
    suite = suite_cls()
    task = suite.get_task(args.task_id)
    initial_states = suite.get_task_init_states(args.task_id)
    if args.trial_id >= len(initial_states):
        raise IndexError(f"trial-id {args.trial_id} out of range for task {args.task_id}")
    initial_state = initial_states[args.trial_id]
    env = make_env(task)
    obs = settle_env(env, initial_state, args.settle_steps)

    records: list[dict[str, Any]] = []
    real_frames: list[np.ndarray] = [rot180(obs["agentview_image"])]
    reference_r = rot180(obs["agentview_image"])
    try:
        for _decision in range(args.max_decisions):
            env_obs = policy_observation(obs, task.language)
            with torch.no_grad():
                action, result = model.predict_action_batch(
                    env_obs=env_obs, mode="train", compute_values=False
                )
            forward_inputs = result["forward_inputs"]
            actions = enforce_signed_gripper(
                action[0, :FUTURE_FRAMES, :ACTION_DIM].detach().cpu().numpy()
            )
            success_flags: list[bool] = []
            env_rewards: list[float] = []
            env_dones: list[bool] = []
            env_infos: list[Any] = []
            for row in actions:
                obs, _reward, _done, _info = env.step(row.tolist())
                reward_array = np.asarray(_reward).reshape(-1)
                done_array = np.asarray(_done).reshape(-1)
                if reward_array.size != 1 or done_array.size != 1:
                    raise ValueError(
                        "standalone LIBERO env must return scalar reward/done, "
                        f"got reward={np.asarray(_reward).shape}, "
                        f"done={np.asarray(_done).shape}"
                    )
                env_reward = float(reward_array[0])
                if not math.isfinite(env_reward):
                    raise ValueError(f"non-finite simulator reward: {env_reward}")
                env_done = bool(done_array[0])
                env_rewards.append(env_reward)
                env_dones.append(env_done)
                env_infos.append(_info)
                success_flags.append(bool(env.check_success()))
                real_frames.append(rot180(obs["agentview_image"]))
            chunk_success = float(any(success_flags))
            chunk_done = bool(chunk_success or any(env_dones))
            records.append(
                {
                    "env_obs": env_obs,
                    "nft_x0": forward_inputs["nft_x0"].detach().cpu(),
                    "reward": chunk_success,
                    "simulator_reward": float(max(env_rewards, default=0.0)),
                    "simulator_rewards": env_rewards,
                    "done": chunk_done,
                    "terminated": chunk_done,
                    "step_dones": env_dones,
                    "step_infos": env_infos,
                    "actions": actions.astype(np.float32),
                }
            )
            if chunk_done:
                break
    finally:
        env.close()

    if not records:
        raise RuntimeError("No decision steps collected")
    instruction = task.language
    n_decisions = len(records)
    full_libero_frames = len(real_frames) - 1
    save_libero_video(
        args.output_dir / "libero_full_trajectory.mp4",
        real_frames[1:],
        f"task {args.task_id} LIBERO",
    )
    num_steps = int(model.config.num_steps)
    schedule = torch.linspace(1, 0, num_steps + 1)

    # ---------------- counterfactual variants ----------------
    # A full 320-step trajectory can contain 40 decisions.  Each Pi05 prefix
    # cache is large, so cache one decision at a time and keep only the sampled
    # x_cur tensors/actions on CPU.  This preserves the full trajectory while
    # avoiding an O(trajectory length) GPU-memory spike.
    variants: list[dict[str, Any]] = [
        {"chunks": [], "xcurs": [], "ts": [], "x0_targets": []}
        for _ in range(args.num_variants)
    ]
    for rec in records:
        ctx = build_model_context(model, rec["env_obs"])
        x0 = None
        try:
            x0 = rec["nft_x0"].to("cuda", torch.float32)  # [1, horizon, dim]
            for k in range(args.num_variants):
                step_idx = int(torch.randint(0, num_steps, (1,)).item())
                t = float(schedule[step_idx].item())
                xcur = (1.0 - t) * x0 + t * torch.randn_like(x0)
                with torch.no_grad():
                    x0_cf = denoise_from(
                        model,
                        ctx,
                        xcur,
                        step_idx,
                        num_steps,
                        noise_method=args.noise_method,
                    )
                    actions_cf = model.output_transform(
                        {"actions": x0_cf, "state": ctx["observation"].state}
                    )["actions"]
                chunk = enforce_signed_gripper(
                    actions_cf[0, :FUTURE_FRAMES, :ACTION_DIM]
                    .detach()
                    .cpu()
                    .numpy()
                )
                variants[k]["chunks"].append(chunk.astype(np.float32))
                variants[k]["xcurs"].append(xcur.detach().cpu())
                variants[k]["ts"].append(t)
                # NFT must regress toward the counterfactual denoised target,
                # not the source trajectory's x0.  Losing this tensor here
                # silently turned every synthetic branch into a copy of the
                # real-stream target during the backward pass.
                variants[k]["x0_targets"].append(x0_cf.detach().cpu())
                del xcur, x0_cf, actions_cf
        finally:
            del ctx
            if x0 is not None:
                del x0
            torch.cuda.empty_cache()

    real_actions = np.concatenate([rec["actions"] for rec in records], axis=0)
    cf_matrix = np.stack(
        [np.concatenate(var["chunks"], axis=0) for var in variants], axis=0
    )  # [K, T, 7]
    pairwise = [
        float(np.linalg.norm(cf_matrix[k] - real_actions, axis=-1).mean())
        for k in range(args.num_variants)
    ]

    # ---------------- Wan open-loop replay (full trajectory per variant) ----------------
    from diffsynth.models.reward_model import TaskEmbedResnetRewModel
    from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline

    rm = (
        TaskEmbedResnetRewModel(
            checkpoint_path=str(args.goal_reward_path),
            task_suite_name=args.task_suite,
        )
        .eval()
        .to("cuda")
    )
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(path=str(args.wan_model_path), offload_device="cpu"),
            ModelConfig(path=str(args.vae_path), offload_device="cpu"),
        ],
    )
    pipe.dit.to("cuda")
    pipe.vae.to("cuda")

    variant_rewards, variant_frames = score_wan_variants(
        pipe,
        rm,
        reference_r,
        instruction,
        variants,
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
        output_dir=args.output_dir,
    )

    # Save the first real-vs-Wan comparison before the policy-gradient pass.
    # This makes the visual artifact available independently of optimization.
    save_video(
        args.output_dir / "real_vs_counterfactual.mp4",
        real_frames[1 : 1 + len(variant_frames[0])],
        variant_frames[0],
        f"task {args.task_id} real vs variant 0",
    )

    # Wan and the reward model occupy most of the GPU.  They are no longer
    # needed after scoring, so release them before constructing NFT graphs for
    # the native Pi05 horizon.
    del pipe, rm
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    # ---------------- losses ----------------
    real_adv = normalize_trajectory_advantages([rec["reward"] for rec in records])
    syn_adv = normalize_group_advantages(variant_rewards)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    optimizer.zero_grad()
    loss_real_sum = 0.0
    loss_syn_sum = 0.0
    total_sum = 0.0
    # Accumulate gradients decision-by-decision.  The scalar division makes
    # this equivalent to the original mean over all decisions/variants while
    # allowing each prefix cache and autograd graph to be released promptly.
    for decision_idx, rec in enumerate(records):
        ctx = build_model_context(model, rec["env_obs"])
        x0 = None
        xcur = None
        loss_real_i = None
        loss_syn_terms_i = []
        loss_syn_i = None
        total_i = None
        try:
            x0 = rec["nft_x0"].to("cuda", torch.float32)
            step_idx = int(torch.randint(0, num_steps, (1,)).item())
            t = float(schedule[step_idx].item())
            xcur = (1.0 - t) * x0 + t * torch.randn_like(x0)
            loss_real_i, _ = nft_loss_for_stream(
                model,
                frozen,
                [ctx],
                [xcur],
                [t],
                [x0],
                [real_adv[decision_idx]],
                form=args.loss_form,
                max_drift=args.max_drift,
                beta=args.beta,
                dpo_beta=args.dpo_beta,
            )
            loss_syn_terms_i = []
            for k in range(args.num_variants):
                xcur_syn = variants[k]["xcurs"][decision_idx].to(
                    "cuda", torch.float32
                )
                loss_k, _ = nft_loss_for_stream(
                    model,
                    frozen,
                    [ctx],
                    [xcur_syn],
                    [variants[k]["ts"][decision_idx]],
                    [
                        variants[k]["x0_targets"][decision_idx].to(
                            "cuda", torch.float32
                        )
                    ],
                    [syn_adv[k]],
                    form=args.loss_form,
                    max_drift=args.max_drift,
                    beta=args.beta,
                    dpo_beta=args.dpo_beta,
                )
                loss_syn_terms_i.append(loss_k)
                del xcur_syn
            loss_syn_i = torch.stack(loss_syn_terms_i).mean()
            total_i = loss_real_i + args.rho * loss_syn_i
            (total_i / n_decisions).backward()
            loss_real_sum += float(loss_real_i.detach().cpu().item())
            loss_syn_sum += float(loss_syn_i.detach().cpu().item())
            total_sum += float(total_i.detach().cpu().item())
        finally:
            del ctx
            if x0 is not None:
                del x0
            if xcur is not None:
                del xcur
            if loss_real_i is not None:
                del loss_real_i
            del loss_syn_terms_i
            if loss_syn_i is not None:
                del loss_syn_i
            if total_i is not None:
                del total_i
            torch.cuda.empty_cache()

    loss_real = torch.tensor(loss_real_sum / n_decisions)
    loss_syn = torch.tensor(loss_syn_sum / n_decisions)
    total = torch.tensor(total_sum / n_decisions)
    grad_norm = float(
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
    )
    optimizer.step()
    model.eval()

    # ---------------- validation & report ----------------
    checks = {
        "loss_finite": bool(math.isfinite(float(total.detach().cpu().item()))),
        "real_adv_in_01": bool(min(real_adv) >= -1e-6 and max(real_adv) <= 1.0 + 1e-6),
        "syn_adv_in_01": bool(min(syn_adv) >= -1e-6 and max(syn_adv) <= 1.0 + 1e-6),
        "variants_actions_finite": bool(np.isfinite(cf_matrix).all()),
        "signed_gripper_range": bool(
            np.logical_and(
                real_actions[:, -1] >= -1.0 - 1e-6,
                real_actions[:, -1] <= 1.0 + 1e-6,
            ).all()
            and np.logical_and(
                cf_matrix[:, :, -1] >= -1.0 - 1e-6,
                cf_matrix[:, :, -1] <= 1.0 + 1e-6,
            ).all()
        ),
        "wan_frames_finite": all(
            np.isfinite(frame).all() for frames in variant_frames for frame in frames
        ),
    }
    reward_values = np.asarray(
        [float(rec["reward"]) for rec in records] + variant_rewards,
        dtype=np.float64,
    )
    reward_signal_available = bool(
        reward_values.size > 0 and np.ptp(reward_values) > 1e-12
    )
    nonzero_gradient = bool(grad_norm > 1e-12)
    report = {
        "task_suite": args.task_suite,
        "task_id": args.task_id,
        "trial_id": args.trial_id,
        "instruction": instruction,
        "pi05_checkpoint": str(args.pi05_path),
        "pi05_norm_stats": str(
            OmegaConf.select(cfg, "rollout.model.openpi_data.norm_stats_path")
            or "checkpoint-native"
        ),
        "action_chunk": FUTURE_FRAMES,
        "max_decisions_requested": args.max_decisions,
        "max_environment_steps_requested": args.max_decisions * FUTURE_FRAMES,
        "pi05_sampling": {
            "num_steps": args.num_steps,
            "noise_method": args.noise_method,
            "noise_level": args.noise_level,
            "num_images_in_input": int(
                OmegaConf.select(cfg, "rollout.model.openpi.num_images_in_input")
            ),
        },
        "gripper_action_contract": "signed_pm1: -1=open, +1=closed",
        "n_decisions": n_decisions,
        "full_libero_frames": full_libero_frames,
        "full_libero_video": str(args.output_dir / "libero_full_trajectory.mp4"),
        "wan_scoring": "full_trajectory_max_frame_reward",
        "wan_frames_per_variant": [len(frames) for frames in variant_frames],
        "synthetic_nft_target": "counterfactual_denoised_x0_cf",
        "real_chunk_rewards": [rec["reward"] for rec in records],
        "real_simulator_rewards": [rec["simulator_reward"] for rec in records],
        "real_adv": real_adv,
        "variant_rewards": variant_rewards,
        "syn_adv": syn_adv,
        "pairwise_l2_vs_real": pairwise,
        "real_gripper_range": [
            float(real_actions[:, -1].min()),
            float(real_actions[:, -1].max()),
        ],
        "counterfactual_gripper_range": [
            float(cf_matrix[:, :, -1].min()),
            float(cf_matrix[:, :, -1].max()),
        ],
        "loss_real": float(loss_real.detach().cpu().item()),
        "loss_syn": float(loss_syn.detach().cpu().item()),
        "loss_total": float(total.detach().cpu().item()),
        "grad_norm": grad_norm,
        "reward_signal_available": reward_signal_available,
        "nonzero_gradient": nonzero_gradient,
        "status": (
            "completed"
            if reward_signal_available and nonzero_gradient
            else "completed_contract_only"
        ),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }
    (args.output_dir / "m0_smoke_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
