#!/usr/bin/env python3
"""Legacy PPO-first diagnostic; not the online WorldSample training path.

The previous M0 used one ``mode=train`` NFT rollout as the anchor for both
streams.  That is useful for checking tensor plumbing, but it can produce an
all-zero reward batch when the exploratory train sampler misses the task.  In
that case the synthetic variants are counterfactuals of a failure and there is
no meaningful positive signal to learn from.

This diagnostic changes the order:

  1. collect real LIBERO rollouts with the regular OpenPI PPO sampling path
     (one SDE denoising step, ``is_nft=False``);
  2. compute chunk-level GAE and apply RLinf's PPO actor+critic loss;
  3. re-roll the updated policy in eval mode to obtain a simulator-grounded
     anchor trajectory;
  4. only then resample that anchor into synthetic action variants and score
     them in LIBERO and, optionally, Wan.

It is intentionally a serial, post-update diagnostic rather than a
replacement for the distributed RLinf runner.  The corrected online path is
``online_ppo_nft_worldsample.py``; keep this file only for historical eval
comparisons and replay debugging.
"""

from __future__ import annotations

import argparse
import gc
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
    ACTION_DIM,
    FUTURE_FRAMES,
    _ensure_openpi_tqdm_compat,
    _ensure_rlinf_ray_compat,
    make_env,
    policy_observation,
    rot180,
    settle_env,
)
from experiments.world_model_grpo.m0_two_stream_smoke import (  # noqa: E402
    build_model_context,
    denoise_from,
    enforce_signed_gripper,
    save_libero_video,
    score_wan_variants,
)
from omegaconf import OmegaConf  # noqa: E402


def _tree_to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _tree_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_tree_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_tree_to_cpu(item) for item in value)
    return value


def _tree_to_device(value: Any, device: str) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _tree_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_tree_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_tree_to_device(item, device) for item in value)
    return value


def _set_model_config(model: Any, key: str, value: Any) -> None:
    """Set a dataclass config field even when the config is frozen."""
    model.config.__dict__[key] = value


def _finite_scalar(value: Any, default: float = 0.0) -> float:
    if torch.is_tensor(value):
        value = value.detach().float().reshape(-1)[0].cpu().item()
    value = float(value)
    return value if math.isfinite(value) else default


def _make_model_config(args: argparse.Namespace) -> Any:
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
    cfg.rollout.model.openpi.noise_method = args.ppo_noise_method
    cfg.rollout.model.openpi.noise_level = args.ppo_noise_level
    # PPO needs a critic.  The GRPO checkpoint has no trained value head, so
    # this serial smoke initializes the head and trains it from simulator
    # returns, exactly as the standard PPO config does.
    cfg.rollout.model.openpi.add_value_head = True
    cfg.rollout.model.openpi.value_after_vlm = True
    cfg.rollout.model.openpi.detach_critic_input = True
    cfg.rollout.model.openpi.is_nft = False
    # RLinf-Pi05-LIBERO-SFT already uses signed gripper statistics.
    OmegaConf.update(
        cfg,
        "rollout.model.openpi_data.output_gripper_action_mode",
        "already_signed",
        force_add=True,
    )
    return cfg


def _collect_episode(
    model: Any,
    task: Any,
    initial_state: np.ndarray,
    *,
    settle_steps: int,
    max_decisions: int,
    mode: str,
    collect_ppo_fields: bool,
    video_path: pathlib.Path,
    collect_online_nft_state: bool = False,
) -> dict[str, Any]:
    """Collect one real LIBERO episode and retain PPO inputs when requested."""
    env = make_env(task)
    obs = settle_env(env, initial_state, settle_steps)
    frames: list[np.ndarray] = [rot180(obs["agentview_image"])]
    records: list[dict[str, Any]] = []
    success = False
    try:
        for _decision in range(max_decisions):
            env_obs = policy_observation(obs, task.language)
            with torch.no_grad():
                action, result = model.predict_action_batch(
                    env_obs=env_obs,
                    mode=mode,
                    compute_values=collect_ppo_fields,
                )
            forward_inputs = result["forward_inputs"]
            model_action = forward_inputs["model_action"]
            model_action = model_action.reshape(
                model_action.shape[0], model.config.action_horizon, model.config.action_dim
            )
            actions = enforce_signed_gripper(
                action[0, :FUTURE_FRAMES, :ACTION_DIM].detach().cpu().numpy()
            )
            success_flags: list[bool] = []
            simulator_rewards: list[float] = []
            simulator_dones: list[bool] = []
            simulator_infos: list[Any] = []
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
                simulator_reward = float(reward_array[0])
                if not math.isfinite(simulator_reward):
                    raise ValueError(
                        f"non-finite simulator reward: {simulator_reward}"
                    )
                simulator_done = bool(done_array[0])
                simulator_rewards.append(simulator_reward)
                simulator_dones.append(simulator_done)
                simulator_infos.append(_info)
                success_flags.append(bool(env.check_success()))
                frames.append(rot180(obs["agentview_image"]))
            chunk_success = float(any(success_flags))
            chunk_done = bool(chunk_success or any(simulator_dones))
            record: dict[str, Any] = {
                "env_obs": _tree_to_cpu(env_obs),
                "x0": model_action.detach().cpu(),
                "actions": actions.astype(np.float32),
                "reward": chunk_success,
                # Keep the raw environment signal for auditability while
                # retaining the task-success reward used by this diagnostic.
                "simulator_reward": float(max(simulator_rewards, default=0.0)),
                "simulator_rewards": simulator_rewards,
                "step_dones": simulator_dones,
                "step_infos": simulator_infos,
                "done": chunk_done,
                "terminated": chunk_done,
                "truncated": False,
            }
            if collect_ppo_fields:
                record.update(
                    {
                        "forward_inputs": _tree_to_cpu(forward_inputs),
                        "prev_logprobs": result["prev_logprobs"].detach().cpu(),
                        "prev_values": result["prev_values"].detach().cpu(),
                        "value": _finite_scalar(result["prev_values"]),
                    }
                )
            if collect_online_nft_state:
                if "nft_x0" not in forward_inputs:
                    raise RuntimeError(
                        "collect_online_nft_state=True but PPO rollout did not "
                        "return forward_inputs['nft_x0']; check the rollout "
                        "model flag"
                    )
                record["nft_x0"] = forward_inputs["nft_x0"].detach().cpu()
            records.append(record)
            if chunk_done:
                success = bool(success or chunk_success)
                break
    finally:
        env.close()

    bootstrap_value: float | None = None
    if records and not records[-1]["terminated"]:
        # Reaching max_decisions is a truncation, not a terminal transition.
        # Bootstrap GAE from the value of the post-chunk observation instead
        # of silently forcing the last transition's next value to zero.
        records[-1]["truncated"] = True
        records[-1]["done"] = True
        if collect_ppo_fields:
            next_env_obs = policy_observation(obs, task.language)
            with torch.no_grad():
                _, bootstrap_result = model.predict_action_batch(
                    env_obs=next_env_obs,
                    mode=mode,
                    compute_values=True,
                )
            bootstrap_value = _finite_scalar(bootstrap_result["prev_values"])

    save_libero_video(video_path, frames[1:], f"{mode} {task.language}")
    return {
        "records": records,
        "frames": frames,
        "success": success,
        "n_decisions": len(records),
        "n_frames": len(frames) - 1,
        "video": str(video_path),
        "bootstrap_value": bootstrap_value,
    }


def _compute_gae(
    episodes: list[dict[str, Any]], gamma: float, gae_lambda: float
) -> tuple[list[float], list[float]]:
    """Compute one scalar chunk advantage for each stored PPO decision."""
    raw_advantages: list[float] = []
    returns: list[float] = []
    for episode in episodes:
        records = episode["records"]
        episode_advantages = [0.0] * len(records)
        gae = 0.0
        for index in reversed(range(len(records))):
            record = records[index]
            value = float(record["value"])
            # ``done`` includes artificial truncation for contract purposes;
            # only a true termination cuts the GAE recursion.  A truncated
            # final chunk uses the separately collected bootstrap value.
            terminated = bool(record.get("terminated", record["done"]))
            if index == len(records) - 1:
                next_value = (
                    0.0
                    if terminated
                    else float(episode.get("bootstrap_value") or 0.0)
                )
            else:
                next_value = float(records[index + 1]["value"])
            delta = float(record["reward"]) + gamma * next_value - value
            gae = delta if terminated else delta + gamma * gae_lambda * gae
            episode_advantages[index] = gae
        raw_advantages.extend(episode_advantages)
        returns.extend(
            float(advantage + record["value"])
            for advantage, record in zip(episode_advantages, records)
        )

    if not raw_advantages:
        return [], []
    array = np.asarray(raw_advantages, dtype=np.float64)
    std = float(array.std())
    if std > 1e-12:
        array = (array - float(array.mean())) / (std + 1e-8)
    else:
        array = array - float(array.mean())
    return [float(item) for item in array], returns


def _ppo_update(
    model: Any,
    episodes: list[dict[str, Any]],
    advantages: list[float],
    returns: list[float],
    *,
    lr: float,
    clip_ratio: float,
    value_clip: float,
    huber_delta: float,
) -> dict[str, Any]:
    """Apply one streaming RLinf actor_critic/PPO update."""
    # Importing the package registers the actor_critic loss implementation.
    __import__("rlinf.algorithms")
    from rlinf.algorithms.registry import policy_loss

    records = [record for episode in episodes for record in episode["records"]]
    if not records:
        return {"updated": False, "reason": "no_records"}
    if not any(float(record["reward"]) > 0.0 for record in records):
        return {
            "updated": False,
            "reason": "no_positive_simulator_reward",
            "n_records": len(records),
        }

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    actor_loss_sum = 0.0
    critic_loss_sum = 0.0
    ratio_sum = 0.0
    n_records = len(records)
    device = str(next(model.parameters()).device)

    for record, advantage, return_value in zip(records, advantages, returns):
        forward_inputs = _tree_to_device(record["forward_inputs"], device)
        old_logprobs = record["prev_logprobs"].to(device=device, dtype=torch.float32)
        prev_values = record["prev_values"].to(device=device, dtype=torch.float32)
        advantages_t = torch.tensor(
            [[float(advantage)]], device=device, dtype=torch.float32
        )
        returns_t = torch.tensor(
            [[float(return_value)]], device=device, dtype=torch.float32
        )
        try:
            output = model(
                forward_inputs=forward_inputs,
                compute_logprobs=True,
                compute_entropy=False,
                compute_values=True,
                use_cache=False,
            )
            current_logprobs = output["logprobs"].float()
            current_values = output["values"].float()
            loss, metrics = policy_loss(
                loss_type="actor_critic",
                logprob_type="chunk_level",
                reward_type="chunk_level",
                task_type="embodied",
                single_action_dim=ACTION_DIM,
                logprobs=current_logprobs,
                values=current_values,
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
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite PPO loss: {loss.item()}")
            (loss / n_records).backward()
            loss_sum += _finite_scalar(loss)
            actor_loss_sum += _finite_scalar(metrics.get("actor/policy_loss", 0.0))
            critic_loss_sum += _finite_scalar(metrics.get("critic/value_loss", 0.0))
            ratio_sum += _finite_scalar(metrics.get("actor/ratio", 1.0), 1.0)
        finally:
            del forward_inputs
            if "output" in locals():
                del output
            if "loss" in locals():
                del loss
            torch.cuda.empty_cache()

    grad_norm = float(
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
    )
    optimizer.step()
    model.eval()
    return {
        "updated": True,
        "n_records": n_records,
        "loss": loss_sum / n_records,
        "actor_loss": actor_loss_sum / n_records,
        "critic_loss": critic_loss_sum / n_records,
        "ratio": ratio_sum / n_records,
        "grad_norm": grad_norm,
        "lr": lr,
    }


def _simulate_chunks(
    task: Any,
    initial_state: np.ndarray,
    chunks: list[np.ndarray],
    *,
    settle_steps: int,
    video_path: pathlib.Path,
    title: str,
) -> dict[str, Any]:
    env = make_env(task)
    obs = settle_env(env, initial_state, settle_steps)
    frames: list[np.ndarray] = [rot180(obs["agentview_image"])]
    success = False
    try:
        for chunk in chunks:
            chunk_done = False
            for row in chunk:
                obs, _reward, _done, _info = env.step(row.tolist())
                done_array = np.asarray(_done).reshape(-1)
                if done_array.size != 1:
                    raise ValueError(
                        "standalone LIBERO env must return scalar done, "
                        f"got {np.asarray(_done).shape}"
                    )
                success = bool(env.check_success()) or success
                chunk_done = bool(done_array[0])
                frames.append(rot180(obs["agentview_image"]))
                if success or chunk_done:
                    break
            if success or chunk_done:
                break
    finally:
        env.close()
    save_libero_video(video_path, frames[1:], title)
    return {
        "success": success,
        "n_frames": len(frames) - 1,
        "video": str(video_path),
    }


def _generate_variants(
    model: Any,
    records: list[dict[str, Any]],
    *,
    num_variants: int,
    num_steps: int,
    noise_method: str,
    seed: int,
) -> list[dict[str, Any]]:
    schedule = torch.linspace(1, 0, num_steps + 1)
    variants: list[dict[str, Any]] = [
        {"chunks": [], "restarts": [], "ts": []}
        for _ in range(num_variants)
    ]
    model.eval()
    for record in records:
        ctx = build_model_context(model, record["env_obs"])
        x0 = record["x0"].to("cuda", torch.float32)
        try:
            for variant_index in range(num_variants):
                local_seed = seed + 1009 * variant_index + len(variants[variant_index]["chunks"])
                random.seed(local_seed)
                np.random.seed(local_seed)
                torch.manual_seed(local_seed)
                torch.cuda.manual_seed_all(local_seed)
                step_index = int(torch.randint(0, num_steps, (1,)).item())
                t = float(schedule[step_index].item())
                xcur = (1.0 - t) * x0 + t * torch.randn_like(x0)
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
                chunk = enforce_signed_gripper(
                    actions_cf[0, :FUTURE_FRAMES, :ACTION_DIM]
                    .detach()
                    .cpu()
                    .numpy()
                )
                variants[variant_index]["chunks"].append(chunk.astype(np.float32))
                variants[variant_index]["restarts"].append(
                    xcur.detach().cpu()
                )
                variants[variant_index]["ts"].append(t)
                del xcur, x0_cf, actions_cf
        finally:
            del ctx, x0
            torch.cuda.empty_cache()
    return variants


def _score_wan(
    args: argparse.Namespace,
    reference_frame: np.ndarray,
    instruction: str,
    variants: list[dict[str, Any]],
    output_dir: pathlib.Path,
) -> tuple[list[float], list[list[np.ndarray]]]:
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
    # Score the same full action sequence that the LIBERO replay above uses.
    # This keeps the optional legacy Wan diagnostic temporally aligned with
    # the synthetic transitions instead of silently scoring only chunk 0.
    rewards, all_frames = score_wan_variants(
        pipe,
        rm,
        reference_frame,
        instruction,
        variants,
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
        output_dir=output_dir,
        output_prefix="wan_variant",
    )
    del pipe, rm
    gc.collect()
    torch.cuda.empty_cache()
    return rewards, all_frames


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
    )
    parser.add_argument("--wan-model-path", type=pathlib.Path, required=True)
    parser.add_argument("--vae-path", type=pathlib.Path, required=True)
    parser.add_argument("--goal-reward-path", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--task-suite", default="libero_goal")
    parser.add_argument("--task-id", type=int, default=7)
    parser.add_argument("--trial-id", type=int, default=9)
    parser.add_argument("--num-real-episodes", type=int, default=4)
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
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--skip-wan", action="store_true")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip PPO update and use the exact eval-mode trajectory as the anchor.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    cfg = _make_model_config(args)
    _ensure_rlinf_ray_compat()
    _ensure_openpi_tqdm_compat()
    from rlinf.models.embodiment.openpi import get_model

    model = get_model(cfg.rollout.model).to("cuda").eval()
    _set_model_config(model, "is_nft", False)

    from libero.libero import benchmark

    suite_cls = benchmark.get_benchmark_dict()[args.task_suite]
    suite = suite_cls()
    task = suite.get_task(args.task_id)
    initial_states = suite.get_task_init_states(args.task_id)
    if args.trial_id >= len(initial_states):
        raise IndexError(f"trial-id {args.trial_id} out of range for task {args.task_id}")

    # ---------------- real PPO data ----------------
    ppo_episodes: list[dict[str, Any]] = []
    for episode_index in range(args.num_real_episodes):
        trial_index = (args.trial_id + episode_index) % len(initial_states)
        episode_seed = args.seed + episode_index
        random.seed(episode_seed)
        np.random.seed(episode_seed)
        torch.manual_seed(episode_seed)
        torch.cuda.manual_seed_all(episode_seed)
        episode = _collect_episode(
            model,
            task,
            initial_states[trial_index],
            settle_steps=args.settle_steps,
            max_decisions=args.max_decisions,
            mode="train",
            collect_ppo_fields=True,
            video_path=args.output_dir / f"ppo_real_episode_{episode_index:02d}.mp4",
        )
        episode["trial_id"] = trial_index
        ppo_episodes.append(episode)
        print(
            json.dumps(
                {
                    "stage": "ppo_rollout",
                    "episode": episode_index,
                    "trial_id": trial_index,
                    "success": episode["success"],
                    "n_decisions": episode["n_decisions"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    all_ppo_records = [
        record for episode in ppo_episodes for record in episode["records"]
    ]
    advantages, returns = _compute_gae(ppo_episodes, args.gamma, args.gae_lambda)
    ppo_update = (
        {"updated": False, "reason": "eval_only"}
        if args.eval_only
        else _ppo_update(
            model,
            ppo_episodes,
            advantages,
            returns,
            lr=args.ppo_lr,
            clip_ratio=args.clip_ratio,
            value_clip=args.value_clip,
            huber_delta=args.huber_delta,
        )
    )

    report: dict[str, Any] = {
        "algorithm_order": "real_PPO_then_synthetic_eval_only",
        "training_semantics": (
            "real_PPO_update_only; synthetic_LIBERO_Wan_replay_has_no_gradient"
        ),
        "task_suite": args.task_suite,
        "task_id": args.task_id,
        "instruction": task.language,
        "pi05_checkpoint": str(args.pi05_path),
        "pi05_norm_stats": str(args.norm_stats_path),
        "action_chunk": FUTURE_FRAMES,
        "ppo_sampling": {
            "mode": "train",
            "is_nft": False,
            "num_steps": args.num_steps,
            "noise_method": args.ppo_noise_method,
            "noise_level": args.ppo_noise_level,
        },
        "ppo": {
            "num_episodes": len(ppo_episodes),
            "num_records": len(all_ppo_records),
            "episode_successes": [bool(episode["success"]) for episode in ppo_episodes],
            "positive_reward_records": int(
                sum(float(record["reward"]) > 0 for record in all_ppo_records)
            ),
            "advantages": advantages,
            "returns": returns,
            "update": ppo_update,
        },
    }

    if not ppo_update.get("updated", False) and not args.eval_only:
        report["status"] = "ppo_update_skipped"
        report["synthetic"] = {
            "status": "not_run",
            "reason": ppo_update.get("reason", "ppo_update_not_applied"),
        }
        (args.output_dir / "m0_ppo_then_synthetic_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    # ---------------- post-PPO real anchor ----------------
    _set_model_config(model, "is_nft", False)
    anchor = _collect_episode(
        model,
        task,
        initial_states[args.trial_id],
        settle_steps=args.settle_steps,
        max_decisions=args.max_decisions,
        mode="eval",
        collect_ppo_fields=False,
        video_path=args.output_dir / "post_ppo_real_anchor.mp4",
    )
    anchor_records = anchor["records"]
    report["post_ppo_anchor"] = {
        "mode": "eval",
        "trial_id": args.trial_id,
        "success": bool(anchor["success"]),
        "n_decisions": anchor["n_decisions"],
        "n_frames": anchor["n_frames"],
        "video": anchor["video"],
    }

    if not anchor_records:
        report["status"] = "anchor_empty"
        report["synthetic"] = {"status": "not_run", "reason": "empty_anchor"}
        (args.output_dir / "m0_ppo_then_synthetic_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    # ---------------- synthetic variants ----------------
    variants = _generate_variants(
        model,
        anchor_records,
        num_variants=args.num_variants,
        num_steps=args.num_steps,
        noise_method=args.synthetic_noise_method,
        seed=args.seed,
    )
    synthetic_sim: list[dict[str, Any]] = []
    for index, variant in enumerate(variants):
        synthetic_sim.append(
            _simulate_chunks(
                task,
                initial_states[args.trial_id],
                variant["chunks"],
                settle_steps=args.settle_steps,
                video_path=args.output_dir / f"synthetic_libero_variant_{index:02d}.mp4",
                title=f"synthetic variant {index}",
            )
        )

    synthetic: dict[str, Any] = {
        "status": "generated",
        "gradient_update": False,
        "num_variants": len(variants),
        "noise_method": args.synthetic_noise_method,
        "sim_success": [bool(item["success"]) for item in synthetic_sim],
        "sim_rollouts": synthetic_sim,
        "first_chunk_pairwise_l2_vs_anchor": [
            float(
                np.linalg.norm(
                    variant["chunks"][0] - anchor_records[0]["actions"], axis=-1
                ).mean()
            )
            for variant in variants
        ],
    }
    if not args.skip_wan:
        reference_frame = anchor["frames"][0]
        wan_rewards, _wan_frames = _score_wan(
            args,
            reference_frame,
            task.language,
            variants,
            args.output_dir,
        )
        synthetic["wan_full_trajectory_rewards"] = wan_rewards
        synthetic["wan_scoring"] = "full_trajectory_max_frame_reward"
    else:
        synthetic["wan_full_trajectory_rewards"] = None

    report["status"] = "completed_eval_anchor_only" if args.eval_only else "completed"
    report["synthetic"] = synthetic
    report["checks"] = {
        "ppo_has_positive_reward": bool(
            report["ppo"]["positive_reward_records"] > 0
        ),
        "post_ppo_anchor_video_exists": pathlib.Path(anchor["video"]).is_file(),
        "synthetic_variant_count": len(variants) == args.num_variants,
        "synthetic_actions_finite": bool(
            all(
                np.isfinite(chunk).all()
                for variant in variants
                for chunk in variant["chunks"]
            )
        ),
    }
    report["all_checks_passed"] = all(
        value for key, value in report["checks"].items() if key != "ppo_has_positive_reward"
    )
    (args.output_dir / "m0_ppo_then_synthetic_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
