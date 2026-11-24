"""Collect complete paired-camera LIBERO truth episodes with a fixed Pi05."""

from __future__ import annotations

import gc
import json
import random
from pathlib import Path

import numpy as np
from wm_update_data import atomic_json, digest, inventory, sha256, write_episode


def collection_plan(
    *,
    policy_config: Path,
    policy_checkpoint: Path,
    per_task: int,
    seed: int,
    max_steps: int,
    settle_steps: int,
    mode: str = "train",
) -> dict:
    """Identify exact weights and protocol, independent of GPU/shard count."""
    checkpoint = Path(policy_checkpoint).resolve()
    if checkpoint.is_dir():
        candidates = [
            checkpoint / "actor/model_state_dict/full_weights.pt",
            checkpoint / "model_state_dict/full_weights.pt",
            checkpoint / "full_weights.pt",
        ]
        checkpoint = next(
            (p for p in candidates if p.is_file()), checkpoint / "model.safetensors"
        )
    if not checkpoint.is_file() or per_task < 1 or max_steps < 8 or settle_steps < 0:
        raise ValueError(
            "Need an existing fixed policy checkpoint and positive collection lengths"
        )
    if mode not in {"train", "eval"}:
        raise ValueError("Collection mode must be train or eval")
    return {
        "suite": "libero_spatial",
        "per_task": per_task,
        "seed": seed,
        "policy_config": str(Path(policy_config).resolve()),
        "policy_config_sha256": sha256(Path(policy_config)),
        "policy_checkpoint": str(checkpoint),
        "policy_sha256": sha256(checkpoint),
        "max_steps": max_steps,
        "settle_steps": settle_steps,
        "mode": mode,
        "execute_actions": 8,
        "predicted_horizon": 10,
        "rgb_size": 256,
        "control_freq": 20,
        "proprio_input": "zero_state_matches_grpo",
    }


def policy_config(plan: dict):
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(plan["policy_config"])
    if cfg.env.train.get("openpi_gripper_action_mode") != "libero_01_to_signed":
        raise ValueError(
            "Expected the current Spatial libero_01_to_signed action boundary"
        )
    model = OmegaConf.create(OmegaConf.to_container(cfg.rollout.model, resolve=True))
    checkpoint = Path(plan["policy_checkpoint"])
    # Native factory recognizes full_weights.pt in the parent model_state_dict.
    model.model_path = str(checkpoint.parent)
    model.openpi.checkpoint_format = "native"
    if int(model.openpi.action_horizon) != 10 or int(model.num_action_chunks) != 8:
        raise ValueError(
            "Collection requires native horizon=10 and eight executed actions"
        )
    if bool(model.get("add_value_head", model.openpi.get("add_value_head", False))):
        raise ValueError("Expected the critic-free GRPO policy")
    return model


def collect_episode(
    env,
    model,
    *,
    initial_state,
    task: str,
    max_steps: int,
    settle_steps: int,
    mode: str,
) -> dict:
    """Record every actual env.step, including a partial final action chunk."""
    import torch

    from experiments.audit_rlinf_pi05_wan_libero_pair import (
        policy_observation,
        proprio_from_obs,
        rot180,
        settle_env,
    )

    obs = settle_env(env, initial_state, settle_steps)
    primary, wrist, states, proprio = [], [], [], []
    actions, successes, rewards, dones = [], [], [], []

    def record_observation():
        primary.append(rot180(obs["agentview_image"]))
        wrist.append(rot180(obs["robot0_eye_in_hand_image"]))
        states.append(
            np.asarray(env.sim.get_state().flatten(), dtype=np.float64).copy()
        )
        proprio.append(proprio_from_obs(obs))

    record_observation()
    while len(actions) < max_steps:
        env_obs = policy_observation(obs, task)
        # Production world-model GRPO uses no physical proprioceptive input.
        env_obs["states"] = torch.zeros_like(env_obs["states"])
        with torch.inference_mode():
            output, _ = model.predict_action_batch(
                env_obs=env_obs, mode=mode, compute_values=False
            )
        chunk = np.asarray(
            output[0, :8, :7].detach().float().cpu(), dtype=np.float32
        ).copy()
        if chunk.shape != (8, 7) or not np.isfinite(chunk).all():
            raise ValueError("Pi05 returned invalid executed action chunk")
        # Same single gripper conversion as the current few-shot GRPO boundary.
        chunk[:, -1] = np.where(chunk[:, -1] >= 0.5, -1.0, 1.0)
        for action in chunk[: max_steps - len(actions)]:
            obs, reward, done, _ = env.step(action.tolist())
            success = bool(env.check_success())
            actions.append(action.copy())
            rewards.append(float(reward))
            dones.append(bool(done))
            successes.append(success)
            record_observation()
            if success or done:
                break
        if successes[-1] or dones[-1]:
            break
    return {
        "primary": primary,
        "wrist": wrist,
        "actions": actions,
        "states": states,
        "proprio": proprio,
        "successes": successes,
        "rewards": rewards,
        "dones": dones,
    }


def collect_shard(args) -> None:
    import torch
    from libero.libero import benchmark

    from experiments.audit_rlinf_pi05_wan_libero_pair import (
        _ensure_openpi_tqdm_compat,
        _ensure_rlinf_ray_compat,
        make_env,
    )

    plan = json.loads((args.output / "collection_plan.json").read_text())
    if sha256(Path(plan["policy_checkpoint"])) != plan["policy_sha256"]:
        raise ValueError("Policy weights changed after collection was planned")
    if sha256(Path(plan["policy_config"])) != plan["policy_config_sha256"]:
        raise ValueError("Policy configuration changed after collection was planned")
    _ensure_openpi_tqdm_compat()
    _ensure_rlinf_ray_compat()
    from rlinf.models.embodiment.openpi import get_model

    model = get_model(policy_config(plan)).to("cuda").eval()
    if bool(model.config.add_value_head):
        raise ValueError(
            "Collector loaded a critic policy instead of the fixed GRPO policy"
        )
    suite = benchmark.get_benchmark_dict()[plan["suite"]]()
    rows = []
    policy_id = plan["policy_sha256"]
    for case in range(10 * plan["per_task"]):
        if case % args.num_shards != args.shard_index:
            continue
        task_id, trial = divmod(case, plan["per_task"])
        task = suite.get_task(task_id)
        initial_states = suite.get_task_init_states(task_id)
        initial_state_id = trial % len(initial_states)
        episode_seed = plan["seed"] + task_id * 100000 + trial
        metadata = {
            "episode_id": digest([digest(plan), task_id, trial]),
            "policy_id": policy_id,
            "suite": plan["suite"],
            "task_id": task_id,
            "task": task.language,
            "trial_id": trial,
            "initial_state_id": initial_state_id,
            "seed": episode_seed,
            "collection_plan_sha256": digest(plan),
        }
        path = args.output / "episodes" / f"task{task_id:02d}_trial{trial:04d}.hdf5"
        if path.exists():
            row = inventory(path)
            if (
                row["trajectory_id"] != metadata["episode_id"]
                or row["policy_id"] != policy_id
            ):
                raise ValueError(
                    f"Existing episode belongs to another collection: {path}"
                )
        else:
            random.seed(episode_seed)
            np.random.seed(episode_seed)
            torch.manual_seed(episode_seed)
            torch.cuda.manual_seed_all(episode_seed)
            env = make_env(task)
            try:
                env.seed(episode_seed)
                data = collect_episode(
                    env,
                    model,
                    initial_state=initial_states[initial_state_id],
                    task=task.language,
                    max_steps=plan["max_steps"],
                    settle_steps=plan["settle_steps"],
                    mode=plan["mode"],
                )
            finally:
                env.close()
                del env
                # Finish cyclic simulator/EGL cleanup before opening data files.
                gc.collect()
            metadata.update(
                success=bool(any(data["successes"])),
                truncated=not any(data["successes"]) and not any(data["dones"]),
                executed_actions=len(data["actions"]),
            )
            write_episode(path, metadata=metadata, **data)
            row = inventory(path)
        rows.append(row)
        atomic_json(
            args.output / "shards" / f"shard_{args.shard_index}.json",
            {
                "plan_sha256": digest(plan),
                "num_shards": args.num_shards,
                "shard_index": args.shard_index,
                "episodes": rows,
            },
        )
        print(
            f"collect shard={args.shard_index}: {len(rows)} task={task_id} trial={trial} "
            f"success={row['success']} actions={row['actions']}",
            flush=True,
        )


def merge_collection(output: Path, num_shards: int) -> dict:
    plan = json.loads((output / "collection_plan.json").read_text())
    rows = []
    for rank in range(num_shards):
        shard = json.loads((output / "shards" / f"shard_{rank}.json").read_text())
        if shard["plan_sha256"] != digest(plan) or shard["num_shards"] != num_shards:
            raise ValueError("Cannot merge different collection plans")
        rows.extend(shard["episodes"])
    if len(rows) != 10 * plan["per_task"] or len(
        {r["trajectory_id"] for r in rows}
    ) != len(rows):
        raise ValueError("Collection incomplete or contains duplicate episodes")
    result = {
        "plan": plan,
        "plan_sha256": digest(plan),
        "episodes": sorted(rows, key=lambda r: r["path"]),
        "summary": {
            "episodes": len(rows),
            "success": sum(r["success"] for r in rows),
            "failure": sum(not r["success"] for r in rows),
        },
    }
    atomic_json(output / "collection.json", result)
    return result
