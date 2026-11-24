"""CPU integration probe for real KIR resets, with stub video/reward models.

Uses the actual WanEnv constructor, reset, observation and chunk-step code.
This validates data/control flow, not learned generation or GPU optimization.
"""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image


class StubWanPipeline:
    """Supply deterministic novel frames while recording model inputs."""

    def __init__(self):
        self.dit = torch.nn.Identity()
        self.vae = torch.nn.Identity()
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(
            {
                "actions": kwargs["action"].detach().cpu().numpy().copy(),
                "references": np.stack(
                    [np.asarray(image) for image in kwargs["input_image"]]
                ),
                "memories": np.stack(
                    [
                        [np.asarray(image) for image in frames]
                        for frames in kwargs["input_image4"]
                    ]
                ),
            }
        )
        return [
            [
                Image.fromarray(np.full((256, 256, 3), 30 + t, np.uint8))
                for t in range(13)
            ]
            for _ in range(kwargs["batch_size"])
        ]

    def generate_noise(self, shape, seed=None, **kwargs):
        """Expose the real pipeline's slot-preserving noise boundary to checks."""
        return torch.randn(shape, generator=torch.Generator().manual_seed(seed or 0))


class StubCosmosClient:
    """Record the production temporal request and return a novel wrist clip."""

    def __init__(self):
        self.calls = []

    def query_temporal_future_batch(self, reference, memory, actions, **kwargs):
        self.calls.append(
            {
                "reference": reference.copy(),
                "memory": memory.copy(),
                "actions": actions.copy(),
                "context_actions": kwargs["context_actions"].copy(),
            }
        )
        clip = np.stack([np.full((224, 224, 3), 90 + t, np.uint8) for t in range(8)])
        return {"future_primary_clip": np.repeat(clip[None], len(actions), axis=0)}

    def close(self):
        pass


class StubReward(torch.nn.Module):
    """Return a fixed nonterminal classifier probability for Resnet & TaskEmbed stubs."""

    def __init__(self, task_suite_name: str = "libero_spatial"):
        super().__init__()
        self.task_suite_name = task_suite_name

    @property
    def net(self):
        return self

    def visual_encoder(self, images):
        return torch.zeros((len(images), 4), device=images.device, dtype=images.dtype)

    def task_embedding(self, task_ids):
        return torch.zeros((len(task_ids), 2), device=task_ids.device, dtype=torch.float32)

    def fusion_layer(self, combined):
        return torch.full((len(combined), 1), 0.25, device=combined.device)

    def forward(self, images):
        return torch.full((len(images), 1), 0.25, device=images.device)


@contextmanager
def cpu_environment(cfg, *, num_envs=16, seed_offset=0, record_metrics=True):
    """Patch only learned models, network transport and the execution device."""
    from rlinf.envs.world_model.world_model_wan_env import WanEnv
    from rlinf.scheduler import Worker

    if cfg.get("shared_wan_pool", {}).get("enabled", False):
        # This fixture replaces learned generation locally and launches no
        # actor. Shared RPC/trajectory equivalence has a separate CPU test.
        cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        cfg.shared_wan_pool.enabled = False

    def connect(env):
        env.cosmos_client = StubCosmosClient()
        env.cosmos_batch_size = 2
        env.cosmos_use_predicted_proprio = False
        env.cosmos_actions_are_normalized = False
        env.cosmos_gripper_action_mode = "already_cosmos"
        env.cosmos_future_denoising_steps = 5

    with (
        patch.object(Worker, "torch_device_type", "cpu"),
        patch.object(WanEnv, "_build_pipeline", lambda env: StubWanPipeline()),
        patch.object(
            WanEnv,
            "_load_reward_model",
            lambda env: StubReward(str(env.cfg.get("task_suite_name", "libero_spatial"))),
        ),
        patch.object(WanEnv, "_build_cosmos_wrist_client", connect),
    ):
        env = WanEnv(
            cfg,
            num_envs,
            seed_offset,
            total_num_processes=8,
            record_metrics=record_metrics,
        )
        try:
            yield env
        finally:
            env.close()


def check_first_success_mask(cfg) -> dict:
    """Exercise terminal transport, native actor masks, GRPO and CPU gradients."""
    from rlinf.algorithms.registry import calculate_adv_and_returns, policy_loss
    from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor

    if cfg.max_episode_steps < 16:
        return {"skipped": "The two-chunk diagnostic needs at least 16 episode steps"}
    threshold = float(cfg.success_reward_threshold)
    run = int(cfg.get("success_consecutive_frames", 1))
    if threshold <= 0 or not 1 <= run <= 8:
        return {
            "skipped": "The diagnostic needs threshold>0 and confirmation length 1..8"
        }
    probabilities = torch.zeros(2, 8, 8)
    # First/middle/last-step successes, second-chunk success, exact threshold,
    # a transient success followed by low scores, and two failures.
    expected_lengths = torch.tensor(
        [run, min(run + 2, 8), 8, 8 + run, 16, min(run + 1, 8), min(run + 3, 8), 16]
    )
    succeeded = torch.tensor([True, True, True, True, False, True, True, False])
    for slot in torch.where(succeeded)[0].tolist():
        step = int(expected_lengths[slot])
        chunk, end = (step - 1) // 8, (step - 1) % 8 + 1
        probabilities[chunk, slot, end - run : end] = threshold if slot == 5 else 1
    with cpu_environment(cfg, num_envs=8) as env:
        eligible = env.dataset.pool.indices["start_kind"] == 0
        if env.dataset.step_budgets is not None:
            eligible &= env.dataset.step_budgets >= 16
        if not eligible.any():
            return {"skipped": "The two-chunk diagnostic needs a 16-action initial start"}
        initial_id = int(np.flatnonzero(eligible)[0])
        env.reset(episode_indices=np.repeat(initial_id, 8))
        actions = torch.zeros((8, 8, 7))
        actions[:, :, -1] = -1
        rewards, dones = [], [torch.zeros(8, 8, dtype=torch.bool)]
        with patch.object(
            env, "_infer_next_chunk_rewards", side_effect=list(probabilities)
        ):
            for _ in range(2):
                _, reward, terminal, truncated, _ = env.chunk_step(actions)
                rewards.append(reward)
                dones.append(terminal | truncated)
        observed_lengths = env.first_success_mask.valid_steps.clone()
        torch.testing.assert_close(observed_lengths, expected_lengths)
        state = env.first_success_mask.state_dict()

    actor = SimpleNamespace(
        cfg=OmegaConf.create(
            {
                "env": {
                    "train": {
                        "rollout_epoch": 1,
                        "auto_reset": False,
                        "ignore_terminations": False,
                    }
                },
                "algorithm": {"reward_type": "action_level", "filter_rewards": False},
            }
        )
    )
    batch = EmbodiedFSDPActor._process_received_rollout_batch(
        actor,
        {
            "rewards": torch.stack(rewards),
            "dones": torch.stack(dones),
        },
    )
    mask = batch["loss_mask"]
    expected_mask = torch.arange(16)[:, None] < expected_lengths[None]
    actual_mask = mask.transpose(1, 2).reshape(16, 8)
    torch.testing.assert_close(actual_mask, expected_mask)
    torch.testing.assert_close(batch["loss_mask_sum"][0, :, 0], expected_lengths)
    scores = (batch["rewards"] * mask).sum(dim=(0, 2))
    expected_scores = succeeded.float() * (cfg.reward_coef if cfg.use_rel_reward else 1)
    torch.testing.assert_close(scores, expected_scores)
    assert torch.count_nonzero(batch["rewards"][~mask]) == 0
    advantages = calculate_adv_and_returns(
        task_type="embodied",
        adv_type="grpo",
        reward_type="action_level",
        num_action_chunks=8,
        rewards=batch["rewards"],
        dones=batch["dones"],
        loss_mask=mask,
        loss_mask_sum=batch["loss_mask_sum"],
        group_size=8,
        normalize_advantages=False,
    )["advantages"]
    expected_advantages = (expected_scores - expected_scores.mean()) / (
        expected_scores.std() + 1e-6
    )
    torch.testing.assert_close(advantages[0, :, 0], expected_advantages)
    logprobs = torch.zeros(16, 8, 7, requires_grad=True)
    loss, _ = policy_loss(
        loss_type="actor",
        task_type="embodied",
        logprob_type="token_level",
        reward_type="action_level",
        single_action_dim=7,
        logprobs=logprobs,
        old_logprobs=torch.zeros_like(logprobs),
        advantages=advantages.reshape(16, 8),
        loss_mask=mask.reshape(16, 8),
        loss_mask_sum=batch["loss_mask_sum"].reshape(16, 8),
        max_episode_steps=16,
        clip_ratio_low=0.2,
        clip_ratio_high=0.28,
    )
    loss.backward()
    gradients = logprobs.grad.reshape(2, 8, 8, 7)
    assert torch.count_nonzero(gradients[~mask]) == 0
    torch.testing.assert_close(
        gradients.sum(dim=(0, 2, 3)),
        -expected_advantages / 8,
    )
    return {
        "enabled": True,
        "device": "cpu",
        "threshold": threshold,
        "real_environment_and_native_actor": True,
        "real_grpo_advantage_and_clipped_loss_backward": True,
        "valid_steps": observed_lengths.tolist(),
        "actor_valid_steps": batch["loss_mask_sum"][0, :, 0].tolist(),
        "first_success_steps": state["first_success_steps"].tolist(),
        "scores": scores.tolist(),
        "post_success_rewards_zero": True,
        "post_success_gradients_zero": True,
        "each_trajectory_normalized_by_its_valid_length": True,
        "learned_video_reward_models_and_gpu_update_tested": False,
    }


def check_resets(cfg, output: Path) -> dict:
    """Check initial/intermediate resets, one chunk and native reset RNG."""
    from rlinf.envs.utils import get_env_attr
    from rlinf.envs.wrappers import RecordVideo

    with cpu_environment(cfg) as env:
        pool = env.dataset.pool
        kinds = pool.indices["start_kind"]
        chosen = {}
        for kind in (0, 1):
            if kind == 1 and env.dataset.kir_fraction == 0:
                continue
            eligible = kinds == kind
            if env.dataset.step_budgets is not None:
                eligible &= env.dataset.step_budgets > 0
            eligible = np.flatnonzero(eligible)
            selected = [int(eligible[0]), int(eligible[-1])]
            ids = np.repeat(selected, env.group_size)
            obs, info = env.reset(episode_indices=ids)
            assert obs["main_images"].shape == (16, 256, 256, 3)
            assert obs["wrist_images"].shape == (16, 256, 256, 3)
            assert torch.count_nonzero(obs["states"]) == 0
            assert env.libero_shadow_env is None
            assert env.current_obs.shape == (16, 3, 1, 5, 256, 256)
            assert env.wrist_memory_model.shape == (16, 4, 224, 224, 3)
            chosen[str(kind)] = {
                "start_ids": selected,
                "source_frames": info["kir_reset"]["source_frame_indices"][
                    ::8
                ].tolist(),
            }
            if env.dataset.step_budgets is not None:
                chosen[str(kind)]["remaining_action_steps"] = env.first_success_mask.step_limits[::8].tolist()
            if kind == 1 and "start_event_t" in pool.indices:
                event_frames = pool.indices["start_event_t"][selected]
                distances = event_frames - pool.indices["start_t"][selected]
                assert np.all(
                    distances == pool.manifest["selection"]["backtrack_frames"]
                )
                chosen[str(kind)]["first_close_action_frames"] = event_frames.tolist()
                chosen[str(kind)]["frames_before_close"] = distances.tolist()
        reference = env.wrist_reference_model.copy()
        anchor = env.wrist_memory_model[:, 0].copy()
        initial_context = env.condition_action.cpu().numpy().copy()
        actions = torch.zeros((16, 8, 7), dtype=torch.float32)
        actions[:, :, 0] = torch.arange(8) / 100
        actions[:, :, -1] = -1
        obs_list, rewards, terminations, truncations, infos = env.chunk_step(actions)
        np.testing.assert_array_equal(
            env.pipe.calls[0]["actions"][:, :5], initial_context
        )
        np.testing.assert_array_equal(
            env.pipe.calls[0]["actions"][:, 5:], actions.numpy()
        )
        np.testing.assert_array_equal(env.wrist_reference_model, reference)
        np.testing.assert_array_equal(env.wrist_memory_model[:, 0], anchor)
        np.testing.assert_array_equal(
            env.wrist_condition_actions[:, 2:], actions.numpy()[:, -3:]
        )
        assert np.all(env.wrist_images == 97)
        assert rewards.shape == terminations.shape == truncations.shape == (16, 8)
        assert env.elapsed_steps == 8
        assert not terminations.any()
        if env.dataset.step_budgets is None:
            assert bool(truncations[:, -1].all()) == (env.elapsed_steps >= cfg.max_episode_steps)
        else:
            torch.testing.assert_close(truncations[:, -1], env.first_success_mask.step_limits <= env.elapsed_steps)
        assert "shadow_return" not in infos[0]["episode"]
        assert not any(key.startswith("libero_") for key in infos[0]["episode"])
        assert env._pending_debug_video_views["libero_primary"] is None
        assert obs_list[0]["states"].count_nonzero() == 0
        before = env._generator.get_state()
        env.update_reset_state_ids()
        expected = env.reset_state_ids.clone()
        env._generator.set_state(before)
        env.update_reset_state_ids()
        assert torch.equal(env.reset_state_ids, expected)
        assert torch.all(
            env.reset_state_ids.reshape(-1, 8)
            == env.reset_state_ids.reshape(-1, 8)[:, :1]
        )
        # Match EnvWorker's actual wrapper stack before loading GPU models.
        wrapped = RecordVideo(env, cfg.video_cfg)
        try:
            assert callable(get_env_attr(wrapped, "offload"))
            assert callable(get_env_attr(wrapped, "onload"))
            assert callable(get_env_attr(wrapped, "update_reset_state_ids"))
            assert get_env_attr(wrapped, "dataset") is env.dataset
            with patch.object(env, "_clear_accelerator_cache"):
                get_env_attr(wrapped, "offload")()
                assert env._is_offloaded
                get_env_attr(wrapped, "onload")()
                assert not env._is_offloaded
        finally:
            wrapped.close()
        report = {
            "native_video_wrapper_offload_lookup_and_round_trip": True,
            "device": "cpu",
            "stub_video_and_reward_models": True,
            "real_environment_constructor_reset_and_chunk_step": True,
            "gpu_inference_and_grpo_update_tested": False,
            "pool_sha256": pool.manifest["pool_sha256"],
            "selection": pool.manifest.get("selection"),
            "kir_fraction": env.dataset.kir_fraction,
            "samples": chosen,
            "same_start_per_group": True,
            "fixed_references_preserved": True,
            "step_frames": env.elapsed_steps,
        }
    if cfg.get("mask_after_first_success", False):
        report["first_success_mask_check"] = check_first_success_mask(cfg)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check_resets(OmegaConf.load(args.config).env.train, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
