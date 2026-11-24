#!/usr/bin/env python3
"""Online PPO + WorldSample/NFT serial reference implementation.

This is the corrected order for the proposed experiment.  A single update
iteration does the following:

    current policy --PPO train--> real LIBERO trajectory
        -> retain o0, PPO fields and nft_x0
        -> 1x8 re-noise and current-policy denoising
        -> full open-loop Wan trajectory from the same o0
        -> ResNet reward
        -> next iteration consumes the result in
           ``L_PPO(real) + rho * L_NFT(synthetic)``.

The script is intentionally a serial reference path.  It makes the online
ordering executable and inspectable before moving the same request/result
contract into RLinf's Ray workers.  By default the scorer generates Wan
primary futures; an optional strict Cosmos-WoVR wrist adapter can be enabled
with ``--cosmos-wrist-port``.  ``--skip-wan`` validates request creation and
PPO fallback only; it deliberately does not manufacture an NFT reward.  If
Wan/Cosmos misses the exact t+1 slot, the reference performs a PPO-only update
and records the late/drop event rather than silently replaying stale synthetic
data.
"""

from __future__ import annotations

import argparse
import gc
import json
import pathlib
import random
import os
import sys
import threading
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
WORLDPILOT_ROOT = pathlib.Path(
    os.environ.get("WORLDPILOT_ROOT", str(PROJECT_ROOT / "third_party/WorldPilot"))
).expanduser().resolve()
CHECKPOINT_ROOT = pathlib.Path(
    os.environ.get("CHECKPOINT_ROOT", str(PROJECT_ROOT / "checkpoints"))
).expanduser().resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# This reference path is intentionally pinned to the checkout used by the
# active RLinf launcher.  Inserting both RLinf trees at index 0 makes their
# precedence depend on loop order and can silently select an older model
# implementation.  Keep the optional OPD checkout available only for legacy
# imports that explicitly need it.
for _path in (RLINF_ROOT, WAN_ROOT):
    _path_str = str(_path)
    if _path_str in sys.path:
        sys.path.remove(_path_str)
    sys.path.insert(0, _path_str)
# Keep the CLI/contract preflight importable without LIBERO, Ray, or Wan.  The
# heavy standalone collector and world-model helpers are imported inside
# ``main`` only after argparse has handled ``--help`` and runtime validation.
CONDITION_FRAMES = 5
FUTURE_FRAMES = 8
ACTION_DIM = 7


def _assert_rlinf_checkout() -> None:
    import rlinf

    source = pathlib.Path(rlinf.__file__).resolve()
    expected = (RLINF_ROOT / "rlinf").resolve()
    if expected not in source.parents:
        raise RuntimeError(
            "online reference path imported an unexpected rlinf checkout: "
            f"{source}; expected under {expected}"
        )

from experiments.world_model_grpo.online_bridge import (  # noqa: E402
    CounterfactualTransition,
    CounterfactualVariant,
    OnlineSyntheticQueue,
    RealPPOTrajectory,
    RealPPOTransition,
    SyntheticRequest,
    SyntheticResult,
    detach_tree_to_cpu,
    renoise_nft_x0,
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _numpy_frame(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim == 4:
        array = array[0]
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"expected one HWC RGB frame, got {array.shape}")
    return np.ascontiguousarray(array.astype(np.uint8))


def _build_real_contract(
    model: Any,
    episode: dict[str, Any],
    *,
    trajectory_id: str,
    source_step: int,
    weight_version: int,
) -> RealPPOTrajectory:
    """Convert the standalone collector's records into the shared contract."""

    transitions: list[RealPPOTransition] = []
    for record in episode["records"]:
        transitions.append(
            RealPPOTransition(
                observation=record["env_obs"],
                forward_inputs=record["forward_inputs"],
                # The standalone collector stores one executed chunk as
                # [chunk, action_dim]; the RLinf contract is batch-first.
                action=torch.as_tensor(
                    record["actions"], dtype=torch.float32
                ).unsqueeze(0),
                prev_logprobs=record["prev_logprobs"],
                prev_values=record["prev_values"],
                reward=record["reward"],
                done=record["done"],
            )
        )
    source = RealPPOTrajectory(
        trajectory_id=trajectory_id,
        source_step=source_step,
        weight_version=weight_version,
        initial_observation=transitions[0].observation if transitions else None,
        transitions=tuple(transitions),
        metadata={"success": bool(episode["success"])},
    )
    source.validate(
        require_nft_state=True,
        action_chunk=FUTURE_FRAMES,
        action_dim=ACTION_DIM,
        model_action_horizon=int(getattr(model.config, "action_horizon", 0)) or None,
        model_action_dim=int(getattr(model.config, "action_dim", 0)) or None,
    )
    return source


def _make_synthetic_request(
    model: Any,
    source: RealPPOTrajectory,
    *,
    num_variants: int,
    num_steps: int,
    noise_method: str,
    seed: int,
    output_dir: pathlib.Path,
    instruction: str,
) -> SyntheticRequest:
    """Immediately branch the exact real trajectory into 1xK variants."""

    request_output_dir = output_dir / source.trajectory_id
    request_output_dir.mkdir(parents=True, exist_ok=True)

    variants: list[list[CounterfactualTransition]] = [
        [] for _ in range(num_variants)
    ]
    actions_by_variant: list[list[torch.Tensor]] = [
        [] for _ in range(num_variants)
    ]
    model.eval()
    for transition_index, transition in enumerate(source.transitions):
        ctx = build_model_context(model, transition.observation)
        x0 = transition.forward_inputs["nft_x0"].to(
            device=next(model.parameters()).device, dtype=torch.float32
        )
        try:
            generator = torch.Generator(device=x0.device)
            generator.manual_seed(seed + 1009 * transition_index)
            xcur_all, step_indices, timesteps = renoise_nft_x0(
                x0,
                num_steps=num_steps,
                num_variants=num_variants,
                generator=generator,
            )
            for variant_index in range(num_variants):
                step_index = int(step_indices[variant_index].item())
                xcur = xcur_all[variant_index : variant_index + 1]
                with torch.no_grad():
                    x0_cf = denoise_from(
                        model,
                        ctx,
                        xcur,
                        step_index,
                        num_steps,
                        noise_method=noise_method,
                    )
                    actions_cf = model.output_transform(
                        {"actions": x0_cf, "state": ctx["observation"].state}
                    )["actions"]
                action_chunk = enforce_signed_gripper(
                    actions_cf[0, :FUTURE_FRAMES, :ACTION_DIM]
                    .detach()
                    .cpu()
                    .numpy()
                )
                synthetic_forward_inputs = detach_tree_to_cpu(
                    transition.forward_inputs
                )
                synthetic_forward_inputs.update(
                    {
                        "nft_xcur": xcur.detach().cpu(),
                        "nft_x0_target": x0_cf.detach().cpu(),
                        "nft_step_index": torch.tensor([step_index], dtype=torch.long),
                        "nft_timestep": torch.tensor(
                            [float(timesteps[variant_index].item())],
                            dtype=torch.float32,
                        ),
                        "nft_noise_level": torch.tensor(
                            [
                                float(
                                    model._get_noise_level(
                                        device=xcur.device,
                                        dtype=xcur.dtype,
                                        sample_method=noise_method,
                                    ).item()
                                )
                            ],
                            dtype=torch.float32,
                        ),
                    }
                )
                with torch.no_grad():
                    v_old = velocity_at(
                        model,
                        ctx,
                        xcur,
                        float(timesteps[variant_index].item()),
                    )
                synthetic_forward_inputs["nft_v_old"] = v_old.detach().cpu()
                variants[variant_index].append(
                    CounterfactualTransition(
                        transition_index=transition_index,
                        observation=transition.observation,
                        nft_xcur=xcur.detach().cpu(),
                        x0_target=x0_cf.detach().cpu(),
                        nft_step_index=step_index,
                        timestep=float(timesteps[variant_index].item()),
                        actions=torch.from_numpy(action_chunk[None]),
                        forward_inputs=synthetic_forward_inputs,
                    )
                )
                actions_by_variant[variant_index].append(
                    torch.from_numpy(action_chunk)
                )
                del xcur, x0_cf, actions_cf
        finally:
            del ctx, x0
            torch.cuda.empty_cache()

    branches: list[CounterfactualVariant] = []
    for variant_index in range(num_variants):
        branch = CounterfactualVariant(
            variant_index=variant_index,
            transitions=tuple(variants[variant_index]),
            action_sequence=torch.stack(actions_by_variant[variant_index], dim=0),
        )
        branch.validate(
            action_chunk=FUTURE_FRAMES,
            action_dim=ACTION_DIM,
            model_action_horizon=int(getattr(model.config, "action_horizon", 0)) or None,
            model_action_dim=int(getattr(model.config, "action_dim", 0)) or None,
        )
        branches.append(branch)

    reference_frame = _numpy_frame(source.initial_observation["main_images"])
    request = SyntheticRequest(
        request_id=f"{source.trajectory_id}:synthetic",
        trajectory_id=source.trajectory_id,
        source_step=source.source_step,
        weight_version=source.weight_version,
        initial_observation=source.initial_observation,
        variants=tuple(branches),
        metadata={
            "instruction": instruction,
            "reference_frame": reference_frame,
            "output_dir": str(request_output_dir),
            "num_steps": num_steps,
            "noise_method": noise_method,
        },
    )
    request.validate(
        expected_variants=num_variants,
        action_chunk=FUTURE_FRAMES,
        action_dim=ACTION_DIM,
        model_action_horizon=int(getattr(model.config, "action_horizon", 0)) or None,
        model_action_dim=int(getattr(model.config, "action_dim", 0)) or None,
    )
    return request


class WanTrajectoryScorer:
    """Wan scorer with an optional strict Cosmos-WoVR wrist side stream."""

    def __init__(self, args: argparse.Namespace) -> None:
        from diffsynth.models.reward_model import TaskEmbedResnetRewModel
        from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline

        self.args = args
        self.reward_model = (
            TaskEmbedResnetRewModel(
                checkpoint_path=str(args.goal_reward_path),
                task_suite_name=args.task_suite,
            )
            .eval()
            .to("cuda")
        )
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            model_configs=[
                ModelConfig(path=str(args.wan_model_path), offload_device="cpu"),
                ModelConfig(path=str(args.vae_path), offload_device="cpu"),
            ],
        )
        self.pipe.dit.to("cuda")
        self.pipe.vae.to("cuda")
        self.cosmos_wrist_adapter = None
        cosmos_wrist_port = getattr(args, "cosmos_wrist_port", None)
        if cosmos_wrist_port is not None:
            from experiments.world_model_grpo.cosmos_wrist_adapter import (
                CosmosWristAdapter,
                CosmosWristAdapterConfig,
            )

            self.cosmos_wrist_adapter = CosmosWristAdapter(
                CosmosWristAdapterConfig(
                    host=getattr(args, "cosmos_wrist_host", "127.0.0.1"),
                    port=cosmos_wrist_port,
                    worldpilot_root=getattr(
                        args, "cosmos_wrist_worldpilot_root", WORLDPILOT_ROOT
                    ),
                    primary_reference=getattr(
                        args, "cosmos_wrist_primary_reference", True
                    ),
                    cross_view_primary_history=(
                        getattr(args, "cosmos_wrist_cross_view_primary_history", False)
                    ),
                    guidance=getattr(args, "cosmos_wrist_guidance", 1.0),
                    future_denoising_steps=getattr(
                        args, "cosmos_wrist_future_denoising_steps", 5
                    ),
                    actions_are_normalized=getattr(
                        args, "cosmos_wrist_actions_are_normalized", False
                    ),
                    gripper_action_mode=getattr(
                        args, "cosmos_wrist_gripper_action_mode", "already_cosmos"
                    ),
                )
            )

    def __call__(self, request: SyntheticRequest) -> SyntheticResult:
        reference = np.asarray(request.metadata["reference_frame"], dtype=np.uint8)
        instruction = str(request.metadata["instruction"])
        output_dir = pathlib.Path(str(request.metadata["output_dir"]))
        sentinel = np.zeros((CONDITION_FRAMES, ACTION_DIM), dtype=np.float32)
        sentinel[:, -1] = -1.0
        rewards: list[float] = []
        num_variants = len(request.variants)
        memories = [np.repeat(reference[None], 4, axis=0) for _ in request.variants]
        # KIR/WanEnv conditions the rolling image memory with the previous
        # four executed actions.  The first chunk has only sentinel/open
        # actions; every later chunk must carry the same action history as the
        # four generated frames in ``memories``.  Rebuilding sentinels for
        # every chunk would make the second chunk's image/action histories
        # inconsistent.
        condition_actions = np.repeat(sentinel[None], num_variants, axis=0)
        frames_by_variant: list[list[np.ndarray]] = [
            [] for _ in request.variants
        ]
        wrist_reference = None
        wrist_frames_by_variant: list[list[np.ndarray]] = [
            [] for _ in request.variants
        ]
        wrist_state = None
        if self.cosmos_wrist_adapter is not None:
            if request.initial_observation is None:
                raise ValueError(
                    "Cosmos wrist scoring requires the real initial observation"
                )
            if "wrist_images" not in request.initial_observation:
                raise ValueError(
                    "Cosmos wrist scoring requires initial_observation['wrist_images']"
                )
            wrist_reference = _numpy_frame(
                request.initial_observation["wrist_images"]
            )
            wrist_state = self.cosmos_wrist_adapter.initial_state(
                reference,
                wrist_reference,
                batch_size=num_variants,
            )
        num_chunks = len(request.variants[0].action_sequence)
        for chunk_index in range(num_chunks):
            current_action_chunks = np.stack(
                [
                    branch.action_sequence[chunk_index].numpy()
                    for branch in request.variants
                ],
                axis=0,
            )
            # Keep the Cosmos query before the Wan query, just like the native
            # hybrid environment: both models see the same pre-chunk state.
            # In cross-view mode, the primary history is updated only after
            # this query, so no future Wan frame leaks into the wrist input.
            wrist_generated_batch = None
            if self.cosmos_wrist_adapter is not None:
                wrist_generated_batch = self.cosmos_wrist_adapter.query_chunk(
                    wrist_state,
                    current_action_chunks,
                    languages=[instruction] * num_variants,
                    batch_seed=int(self.args.seed + 1000 * chunk_index),
                )
            batch_actions = np.stack(
                [
                    np.concatenate(
                        [
                            condition_actions[variant_index],
                            current_action_chunks[variant_index],
                        ],
                        axis=0,
                    )
                    for variant_index, branch in enumerate(request.variants)
                ],
                axis=0,
            )
            # Match WanEnv's native KIR contract: input_image is the fixed
            # real o0 reference, while input_image4 is rolling generated
            # memory. Reusing o0 is intentional anchor conditioning.
            with torch.inference_mode(), torch.amp.autocast(
                "cuda", dtype=torch.bfloat16
            ):
                generated = self.pipe(
                    seed=int(self.args.seed + 1000 * chunk_index),
                    tiled=False,
                    input_image=[_numpy_to_pil(reference)] * num_variants,
                    input_image4=[
                        [_numpy_to_pil(frame) for frame in memory]
                        for memory in memories
                    ],
                    action=torch.from_numpy(batch_actions).to("cuda", torch.float32),
                    height=256,
                    width=256,
                    num_frames=CONDITION_FRAMES + FUTURE_FRAMES,
                    num_inference_steps=self.args.num_inference_steps,
                    cfg_scale=1.0,
                    progress_bar_cmd=lambda values: values,
                    batch_size=num_variants,
                )
            generated_batch = output_to_uint8(generated)[:, CONDITION_FRAMES:]
            del generated
            if self.cosmos_wrist_adapter is not None:
                if wrist_generated_batch is None or wrist_state is None:
                    raise AssertionError("Cosmos wrist state was not initialized")
                if (
                    self.cosmos_wrist_adapter.config.cross_view_primary_history
                ):
                    self.cosmos_wrist_adapter.update_primary_history(
                        wrist_state, generated_batch
                    )
            for variant_index in range(num_variants):
                generated_chunk = [
                    np.ascontiguousarray(frame)
                    for frame in generated_batch[variant_index]
                ]
                frames_by_variant[variant_index].extend(generated_chunk)
                memories[variant_index] = np.stack(generated_chunk[-4:], axis=0)
                condition_actions[variant_index, 1:, :] = (
                    current_action_chunks[variant_index][-4:]
                )
                if wrist_generated_batch is not None:
                    wrist_frames_by_variant[variant_index].extend(
                        np.ascontiguousarray(frame)
                        for frame in wrist_generated_batch[variant_index]
                    )

        for variant_index, (branch, generated_frames) in enumerate(
            zip(request.variants, frames_by_variant)
        ):
            reward = score_frames(self.reward_model, generated_frames, instruction)
            rewards.append(float(reward))
            target = np.repeat(reference[None], len(generated_frames), axis=0)
            save_video(
                output_dir / f"wan_variant_{branch.variant_index:02d}_full.mp4",
                target,
                np.stack(generated_frames, axis=0),
                f"Wan full synthetic variant {branch.variant_index}",
            )
            if wrist_reference is not None:
                wrist_frames = wrist_frames_by_variant[variant_index]
                if len(wrist_frames) != len(generated_frames):
                    raise RuntimeError(
                        "Wan/Cosmos generated horizons diverged: "
                        f"wan={len(generated_frames)}, wrist={len(wrist_frames)}"
                    )
                save_video(
                    output_dir
                    / f"cosmos_wrist_variant_{branch.variant_index:02d}_full.mp4",
                    np.repeat(wrist_reference[None], len(wrist_frames), axis=0),
                    np.stack(wrist_frames, axis=0),
                    f"Cosmos wrist full synthetic variant {branch.variant_index}",
                )
        world_model_name = "wan_primary_full_trajectory"
        metadata: dict[str, Any] = {
            "trajectory_scored": True,
            "reward_aggregation": "max_over_generated_frames",
        }
        if self.cosmos_wrist_adapter is not None:
            world_model_name = "wan_primary_full_trajectory+cosmos_wrist_wovr"
            metadata.update(
                {
                    "cosmos_wrist": True,
                    "cosmos_metadata": dict(self.cosmos_wrist_adapter.metadata),
                    "cosmos_wrist_primary_reference": bool(
                        self.cosmos_wrist_adapter.config.primary_reference
                    ),
                    "cosmos_wrist_cross_view_primary_history": bool(
                        self.cosmos_wrist_adapter.config.cross_view_primary_history
                    ),
                }
            )
        return SyntheticResult(
            request_id=request.request_id,
            trajectory_id=request.trajectory_id,
            source_step=request.source_step,
            weight_version=request.weight_version,
            rewards=tuple(rewards),
            wm_name=world_model_name,
            metadata=metadata,
        )

    def close(self) -> None:
        if self.cosmos_wrist_adapter is not None:
            self.cosmos_wrist_adapter.close()
        del self.pipe, self.reward_model
        gc.collect()
        torch.cuda.empty_cache()


def _numpy_to_pil(frame: np.ndarray) -> Any:
    from PIL import Image

    return Image.fromarray(np.asarray(frame, dtype=np.uint8))


def _paired_update(
    model: Any,
    episodes: list[dict[str, Any]],
    advantages: list[float],
    returns: list[float],
    synthetic_pairs: list[tuple[SyntheticRequest, SyntheticResult]],
    *,
    optimizer: torch.optim.Optimizer,
    rho: float,
    clip_ratio: float,
    value_clip: float,
    huber_delta: float,
    max_drift: float,
    beta: float,
    dpo_beta: float,
    loss_form: str,
    nft_sum_type: str,
    nft_weight_mode: str,
    nft_weight_scale: list[float],
) -> dict[str, Any]:
    """Apply one paired update, with real PPO and lagged synthetic NFT."""

    from rlinf.algorithms.registry import policy_loss

    records = [record for episode in episodes for record in episode["records"]]
    if not records:
        return {"updated": False, "reason": "no_real_records"}
    if len(records) != len(advantages) or len(records) != len(returns):
        raise ValueError("PPO records/advantages/returns length mismatch")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    ppo_loss_sum = 0.0
    nft_loss_sum = 0.0
    n_ppo = len(records)
    device = str(next(model.parameters()).device)

    for record, advantage, return_value in zip(records, advantages, returns):
        forward_inputs = _tree_to_device(record["forward_inputs"], device)
        old_logprobs = record["prev_logprobs"].to(device=device, dtype=torch.float32)
        prev_values = record["prev_values"].to(device=device, dtype=torch.float32)
        advantages_t = torch.tensor([[float(advantage)]], device=device)
        returns_t = torch.tensor([[float(return_value)]], device=device)
        output = model(
            forward_inputs=forward_inputs,
            compute_logprobs=True,
            compute_entropy=False,
            compute_values=True,
            use_cache=False,
        )
        ppo_loss, _metrics = policy_loss(
            loss_type="actor_critic",
            logprob_type="chunk_level",
            reward_type="chunk_level",
            task_type="embodied",
            single_action_dim=ACTION_DIM,
            logprobs=output["logprobs"],
            values=output["values"],
            old_logprobs=old_logprobs,
            advantages=advantages_t,
            returns=returns_t,
            prev_values=prev_values,
            clip_ratio_high=clip_ratio,
            clip_ratio_low=clip_ratio,
            value_clip=value_clip,
            huber_delta=huber_delta,
            loss_mask=None,
            loss_mask_sum=None,
            max_episode_steps=None,
            critic_warmup=False,
        )
        if not torch.isfinite(ppo_loss):
            raise FloatingPointError(f"non-finite PPO loss: {ppo_loss.item()}")
        (ppo_loss / n_ppo).backward()
        ppo_loss_sum += _finite_scalar(ppo_loss)
        del forward_inputs, output, ppo_loss

    syn_loss_count = 0
    for request, result in synthetic_pairs:
        if not result.is_success:
            continue
        synthetic_advantages = normalize_group_advantages(list(result.rewards))
        branch_loss_sum = 0.0
        request_transition_count = len(request.variants[0].transitions)
        for branch, branch_advantage in zip(
            request.variants, synthetic_advantages
        ):
            n_transitions = len(branch.transitions)
            if n_transitions == 0:
                continue
            for transition in branch.transitions:
                context = build_model_context(model, transition.observation)
                transition_loss, _ = nft_loss_for_stream(
                    model,
                    None,
                    [context],
                    [transition.nft_xcur.to(device=device, dtype=torch.float32)],
                    [float(transition.timestep)],
                    [transition.x0_target.to(device=device, dtype=torch.float32)],
                    [float(branch_advantage)],
                    form=loss_form,
                    max_drift=max_drift,
                    beta=beta,
                    dpo_beta=dpo_beta,
                    v_old_list=[
                        transition.forward_inputs["nft_v_old"].to(
                            device=device, dtype=torch.float32
                        )
                    ],
                    noise_level_list=[
                        float(
                            transition.forward_inputs["nft_noise_level"]
                            .reshape(-1)[0]
                            .item()
                        )
                    ],
                    step_index_list=[
                        int(
                            transition.forward_inputs["nft_step_index"]
                            .reshape(-1)[0]
                            .item()
                        )
                    ],
                    num_steps=int(request.metadata["num_steps"]),
                    nft_weight_mode=nft_weight_mode,
                    nft_weight_scale=tuple(nft_weight_scale),
                    nft_sum_type=nft_sum_type,
                )
                if not torch.isfinite(transition_loss):
                    raise FloatingPointError("non-finite synthetic NFT loss")
                # Backward one transition at a time so prefix caches and
                # synthetic graphs do not accumulate for a full trajectory.
                transition_loss_weight = float(rho) / (
                    len(synthetic_pairs)
                    * len(request.variants)
                    * n_transitions
                )
                (transition_loss * transition_loss_weight).backward()
                branch_loss_sum += _finite_scalar(transition_loss)
                del context, transition_loss
        if request.variants:
            # ``branch_loss_sum`` contains one scalar per transition.  The
            # backward path above already averages by branch and transition;
            # the reported metric must use the same normalization or it is
            # inflated by the number of decisions in a trajectory.
            nft_loss_sum += branch_loss_sum / (
                len(request.variants) * request_transition_count
            )
            syn_loss_count += 1

    grad_norm = float(
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
    )
    optimizer.step()
    model.eval()
    ppo_loss_value = ppo_loss_sum / n_ppo
    nft_loss_value = nft_loss_sum / max(syn_loss_count, 1)
    total_loss = ppo_loss_value + float(rho) * nft_loss_value
    current_lr = float(optimizer.param_groups[0].get("lr", 0.0))
    return {
        "updated": True,
        "n_ppo_records": n_ppo,
        "n_synthetic_requests": syn_loss_count,
        "loss_ppo": ppo_loss_value,
        "loss_nft": nft_loss_value,
        "loss_total": total_loss,
        "rho": float(rho),
        "grad_norm": grad_norm,
        "lr": current_lr,
    }


def _zero_world_model_scorer(request: SyntheticRequest) -> SyntheticResult:
    """Contract-only result; never manufacture a synthetic reward."""

    return SyntheticResult(
        request_id=request.request_id,
        trajectory_id=request.trajectory_id,
        source_step=request.source_step,
        weight_version=request.weight_version,
        rewards=(),
        wm_name="disabled_contract_scorer",
        status="disabled",
        error="world_model_disabled",
        metadata={"world_model_disabled": True},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--pi05-path", type=pathlib.Path, required=True)
    parser.add_argument("--norm-stats-path", type=pathlib.Path, default=None)
    parser.add_argument("--wan-model-path", type=pathlib.Path, default=None)
    parser.add_argument("--vae-path", type=pathlib.Path, default=None)
    parser.add_argument("--goal-reward-path", type=pathlib.Path, default=None)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--task-suite", default="libero_goal")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--trial-id", type=int, default=0)
    parser.add_argument("--num-updates", type=int, default=2)
    parser.add_argument("--episodes-per-update", type=int, default=1)
    parser.add_argument("--settle-steps", type=int, default=15)
    parser.add_argument("--max-decisions", type=int, default=40)
    parser.add_argument("--num-variants", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--ppo-noise-method", choices=("flow_sde", "flow_ode"), default="flow_sde")
    parser.add_argument("--ppo-noise-level", type=float, default=0.3)
    parser.add_argument("--synthetic-noise-method", choices=("flow_sde", "flow_ode"), default="flow_sde")
    parser.add_argument("--num-inference-steps", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ppo-lr", type=float, default=5e-6)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--value-clip", type=float, default=0.2)
    parser.add_argument("--huber-delta", type=float, default=10.0)
    parser.add_argument("--rho", type=float, default=0.3)
    parser.add_argument("--max-drift", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--dpo-beta", type=float, default=1.0)
    parser.add_argument("--loss-form", choices=("mse", "dpo"), default="mse")
    parser.add_argument(
        "--nft-sum-type",
        choices=("action_level", "chunk_level"),
        default="action_level",
        help="NFT energy aggregation; action_level matches RLinf's default worker.",
    )
    parser.add_argument(
        "--nft-weight-mode",
        choices=("constant", "t", "1-t", "inv_t2", "inv_t3", "sigma", "adaptive", "auto"),
        default="auto",
    )
    parser.add_argument(
        "--nft-weight-scale",
        type=float,
        nargs="+",
        default=[1.0, 0.2],
        help="One symmetric scale or [positive_scale negative_scale].",
    )
    parser.add_argument("--max-pending", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--skip-wan", action="store_true")
    parser.add_argument(
        "--cosmos-wrist-port",
        type=int,
        default=None,
        help="Enable the strict Cosmos-WoVR wrist side stream on this port.",
    )
    parser.add_argument("--cosmos-wrist-host", default="127.0.0.1")
    parser.add_argument(
        "--cosmos-wrist-worldpilot-root",
        type=pathlib.Path,
        default=WORLDPILOT_ROOT,
    )
    parser.add_argument(
        "--cosmos-wrist-primary-reference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use the fixed primary-camera reference layout used by the "
            "canonical Goal v3 wrist checkpoint (default: true). Use "
            "--no-cosmos-wrist-primary-reference only for an older "
            "wrist-reference checkpoint."
        ),
    )
    parser.add_argument(
        "--cosmos-wrist-cross-view-primary-history", action="store_true"
    )
    parser.add_argument("--cosmos-wrist-guidance", type=float, default=1.0)
    parser.add_argument(
        "--cosmos-wrist-future-denoising-steps", type=int, default=5
    )
    parser.add_argument(
        "--cosmos-wrist-actions-are-normalized", action="store_true"
    )
    parser.add_argument(
        "--cosmos-wrist-gripper-action-mode",
        choices=("already_cosmos", "libero_01_to_cosmos"),
        default="already_cosmos",
    )
    args = parser.parse_args()

    # Import the environment/model path only for an actual run.  This keeps
    # ``--help`` and config/contract preflight usable in shells that do not
    # have the GPU6 LIBERO dependency closure.
    global _collect_episode, _compute_gae, _finite_scalar, _make_model_config
    global _set_model_config, _tree_to_device, _ensure_openpi_tqdm_compat
    global _ensure_rlinf_ray_compat, output_to_uint8, save_video
    global build_model_context, denoise_from, enforce_signed_gripper
    global nft_loss_for_stream, normalize_group_advantages, score_frames
    global velocity_at
    from experiments.audit_rlinf_pi05_wan_libero_pair import (
        _ensure_openpi_tqdm_compat,
        _ensure_rlinf_ray_compat,
        output_to_uint8,
        save_video,
    )
    from experiments.world_model_grpo.m0_ppo_then_synthetic import (
        _collect_episode,
        _compute_gae,
        _finite_scalar,
        _make_model_config,
        _set_model_config,
        _tree_to_device,
    )
    from experiments.world_model_grpo.m0_two_stream_smoke import (
        build_model_context,
        denoise_from,
        enforce_signed_gripper,
        nft_loss_for_stream,
        normalize_group_advantages,
        score_frames,
        velocity_at,
    )

    if args.num_updates <= 0 or args.episodes_per_update <= 0:
        raise ValueError("num-updates and episodes-per-update must be positive")
    if not args.skip_wan and args.num_updates < 2:
        raise ValueError(
            "Wan mode needs at least two updates: synthetic work from update "
            "t is consumed at t+1; use --skip-wan for a one-update contract smoke"
        )
    if args.max_pending < 2 * args.episodes_per_update:
        raise ValueError(
            "max-pending must hold one lagged batch plus the current batch; "
            f"got {args.max_pending}, need at least {2 * args.episodes_per_update}"
        )
    if len(args.nft_weight_scale) not in (1, 2):
        raise ValueError("--nft-weight-scale expects one value or two values")
    if any(value <= 0.0 or not np.isfinite(value) for value in args.nft_weight_scale):
        raise ValueError("--nft-weight-scale values must be finite and positive")
    if len(args.nft_weight_scale) == 1:
        args.nft_weight_scale = [args.nft_weight_scale[0]] * 2
    if args.skip_wan:
        if args.cosmos_wrist_port is not None:
            raise ValueError(
                "--skip-wan is contract-only and cannot start Cosmos wrist"
            )
        if any(
            value is not None
            for value in (
                args.wan_model_path,
                args.vae_path,
                args.goal_reward_path,
            )
        ):
            raise ValueError("--skip-wan does not accept world-model checkpoints")
    elif any(
        value is None
        for value in (args.wan_model_path, args.vae_path, args.goal_reward_path)
    ):
        raise ValueError(
            "Wan mode requires --wan-model-path, --vae-path and "
            "--goal-reward-path"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _set_seed(args.seed)
    if args.norm_stats_path is None:
        args.norm_stats_path = (
            CHECKPOINT_ROOT
            / "pi05/RLinf-Pi05-LIBERO-SFT/physical-intelligence/libero"
        )

    cfg = _make_model_config(args)
    _ensure_rlinf_ray_compat()
    _ensure_openpi_tqdm_compat()
    _assert_rlinf_checkout()
    from rlinf.models.embodiment.openpi import get_model

    model = get_model(cfg.rollout.model).to("cuda").eval()
    _set_model_config(model, "is_nft", False)
    _set_model_config(model, "collect_online_nft_state", True)
    # Keep Adam's state across paired updates.  Creating the optimizer inside
    # _paired_update would reset moments on every iteration and silently turn
    # the advertised multi-update run into repeated one-step optimizations.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.ppo_lr)
    from libero.libero import benchmark

    suite_cls = benchmark.get_benchmark_dict()[args.task_suite]
    suite = suite_cls()
    task = suite.get_task(args.task_id)
    initial_states = suite.get_task_init_states(args.task_id)
    if args.trial_id >= len(initial_states):
        raise IndexError(f"trial-id {args.trial_id} out of range for task {args.task_id}")

    scorer: Any
    if args.skip_wan:
        scorer = _zero_world_model_scorer
    else:
        scorer = WanTrajectoryScorer(args)
    # The serial reference currently colocates the policy and Wan/Cosmos on
    # one process/device.  A ThreadPoolExecutor alone does not make CUDA
    # modules safe to run concurrently, so guard all policy and scorer GPU
    # work with one process-local lock.  This preserves the queue's lag
    # semantics without allowing silent allocator/race failures.
    gpu_lock = threading.Lock()

    def guarded_scorer(request: SyntheticRequest) -> SyntheticResult:
        with gpu_lock:
            return scorer(request)

    queue = OnlineSyntheticQueue(
        max_pending=args.max_pending,
        min_lag=1,
        max_workers=1,
    )
    pending: dict[str, tuple[SyntheticRequest, SyntheticResult | None]] = {}
    reports: list[dict[str, Any]] = []
    submit_failures: list[dict[str, Any]] = []
    try:
        for update_step in range(args.num_updates):
            # Consume t-1 before submitting t.  This preserves the one-update
            # lag and frees queue capacity before the current real rollout is
            # branched, even when Wan is slower than PPO.
            ready = queue.poll(update_step)
            synthetic_pairs: list[tuple[SyntheticRequest, SyntheticResult]] = []
            for result in ready:
                request, _ = pending.pop(result.request_id, (None, None))
                if request is not None and result.is_success:
                    synthetic_pairs.append((request, result))

            episodes: list[dict[str, Any]] = []
            sources: list[RealPPOTrajectory] = []
            for episode_index in range(args.episodes_per_update):
                absolute_episode = update_step * args.episodes_per_update + episode_index
                trial_index = (args.trial_id + absolute_episode) % len(initial_states)
                episode_seed = args.seed + absolute_episode
                _set_seed(episode_seed)
                with gpu_lock:
                    episode = _collect_episode(
                        model,
                        task,
                        initial_states[trial_index],
                        settle_steps=args.settle_steps,
                        max_decisions=args.max_decisions,
                        mode="train",
                        collect_ppo_fields=True,
                        video_path=args.output_dir
                        / f"ppo_real_update_{update_step:04d}_episode_{episode_index:02d}.mp4",
                        collect_online_nft_state=True,
                    )
                    episodes.append(episode)
                    source = _build_real_contract(
                        model,
                        episode,
                        trajectory_id=f"update-{update_step:04d}-episode-{episode_index:02d}",
                        source_step=update_step,
                        weight_version=update_step,
                    )
                    request = _make_synthetic_request(
                        model,
                        source,
                        num_variants=args.num_variants,
                        num_steps=args.num_steps,
                        noise_method=args.synthetic_noise_method,
                        seed=episode_seed,
                        output_dir=args.output_dir,
                        instruction=task.language,
                    )
                try:
                    queue.submit(request, guarded_scorer)
                except RuntimeError as exc:
                    submit_failures.append(
                        {
                            "request_id": request.request_id,
                            "source_step": request.source_step,
                            "reason": str(exc),
                        }
                    )
                    print(
                        json.dumps(
                            {
                                "stage": "synthetic_submit_skipped",
                                "request_id": request.request_id,
                                "source_step": request.source_step,
                                "reason": str(exc),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                else:
                    pending[request.request_id] = (request, None)
                sources.append(source)
                print(
                    json.dumps(
                        {
                            "stage": "online_real_rollout_and_submit",
                            "update": update_step,
                            "trajectory_id": source.trajectory_id,
                            "source_step": source.source_step,
                            "success": bool(episode["success"]),
                            "decisions": len(source.transitions),
                            "variants": len(request.variants),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

            # Wan/Cosmos may finish while the real simulator rollout is
            # running.  Poll once more at the same exact t+1 slot before the
            # paired update; otherwise a result that became ready during the
            # rollout would be needlessly classified as late at t+2.
            ready_after_rollout = queue.poll(update_step)
            for result in ready_after_rollout:
                request, _ = pending.pop(result.request_id, (None, None))
                if request is not None and result.is_success:
                    synthetic_pairs.append((request, result))

            advantages, returns = _compute_gae(
                episodes, args.gamma, args.gae_lambda
            )
            with gpu_lock:
                update_report = _paired_update(
                    model,
                    episodes,
                    advantages,
                    returns,
                    synthetic_pairs,
                    optimizer=optimizer,
                    rho=args.rho,
                    clip_ratio=args.clip_ratio,
                    value_clip=args.value_clip,
                    huber_delta=args.huber_delta,
                    max_drift=args.max_drift,
                    beta=args.beta,
                    dpo_beta=args.dpo_beta,
                    loss_form=args.loss_form,
                    nft_sum_type=args.nft_sum_type,
                    nft_weight_mode=args.nft_weight_mode,
                    nft_weight_scale=args.nft_weight_scale,
                )
            reports.append(
                {
                    "update": update_step,
                    "real_trajectories": [source.trajectory_id for source in sources],
                    "synthetic_consumed": [result.request_id for _, result in synthetic_pairs],
                    "queue": queue.metrics(),
                    "ppo": update_report,
                }
            )
            print(json.dumps(reports[-1], ensure_ascii=False), flush=True)

        # There is no next PPO update to consume the final lagged jobs.  Drain
        # only to report their completion; do not apply an unpaired update.
        while queue.pending():
            ready = queue.poll(args.num_updates + 1)
            for result in ready:
                pending.pop(result.request_id, None)
            if not ready:
                # The queue is backed by a worker future; a short wait avoids a
                # busy loop while keeping the script responsive to failures.
                import time

                time.sleep(0.1)
        report = {
            "status": "completed_reference_path",
            "algorithm_order": "online_real_ppo_rollout_to_lagged_synthetic_nft",
            "task_suite": args.task_suite,
            "task_id": args.task_id,
            "action_chunk": FUTURE_FRAMES,
            "num_variants": args.num_variants,
            "world_model": (
                "disabled_contract_scorer"
                if args.skip_wan
                else (
                    "wan_primary_full_trajectory+cosmos_wrist_wovr"
                    if args.cosmos_wrist_port is not None
                    else "wan_primary_full_trajectory"
                )
            ),
            "loss": "L_PPO(real) + rho * L_NFT(synthetic)",
            "rho": args.rho,
            "nft_sum_type": args.nft_sum_type,
            "nft_weight_mode": args.nft_weight_mode,
            "nft_weight_scale": args.nft_weight_scale,
            "synthetic_consumption": "exact_lag_1_drop_late",
            "synthetic_submit_failures": submit_failures,
            "updates": reports,
            # ``pending`` is the actor-side provenance map and intentionally
            # keeps late-dropped requests for audit.  It is not the queue's
            # execution state.  Report the latter here; otherwise a healthy
            # exact-lag drain would look as if work were still running merely
            # because a request missed its t+1 consumption slot.
            "pending_after_drain": len(queue.pending()),
            "provenance_records_after_drain": len(pending),
            "queue": queue.metrics(),
        }
        (args.output_dir / "online_ppo_nft_worldsample_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
    finally:
        queue.close(wait=True)
        if hasattr(scorer, "close"):
            scorer.close()


if __name__ == "__main__":
    main()
