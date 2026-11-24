"""CPU production-method checks for compact generation and frozen suffixes.

Wan/Cosmos models are replaced with deterministic, slot-sensitive models.
Environment methods, first-success masks and native actor loss are real.
"""

import ast
import importlib.util
import sys
import unittest
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image

PROJECT = Path(__file__).resolve().parents[2]
WORLD = PROJECT / "third_party/RLinf/rlinf/envs/world_model"
NATIVE = Path("/data41/qianb2608/wan/RLinf_opd/rlinf")


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract_functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    methods = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in methods} != set(names):
        raise AssertionError(f"Missing native functions in {path}")
    for method in methods:
        method.decorator_list = []
    exec(  # noqa: S102 -- execute real source methods with only CPU dependencies
        compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace
    )


def production_environment_class():
    path = WORLD / "world_model_wan_env.py"
    tree = ast.parse(path.read_text())
    original = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "WanEnv"
    )
    names = {
        "_guarded_model_call",
        "_infer_next_chunk_frames",
        "_primary_clip_uint8",
        "_append_wan_frames",
        "_query_cosmos_wrist_wovr",
        "_reset_metrics",
        "_record_metrics",
        "_calc_step_reward",
        "_score_classifier_chunk",
        "_wrap_obs",
        "chunk_step",
    }
    methods = [
        node
        for node in original.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in methods} != names:
        raise AssertionError("Production Wan methods changed")
    # Keep original decorators, including torch.no_grad and staticmethod.
    original.body, original.bases = methods, []
    namespace = {
        "np": np,
        "torch": torch,
        "F": F,
        "Image": Image,
        "nullcontext": nullcontext,
        "convert_gripper_actions_for_cosmos": lambda actions, mode: actions.copy(),
    }
    exec(  # noqa: S102 -- actual environment code, no model/dependency imports
        compile(ast.Module(body=[original], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    cls = namespace["WanEnv"]
    cls.elapsed_steps = property(
        lambda self: self._elapsed_steps,
        lambda self, value: setattr(self, "_elapsed_steps", value),
    )
    return cls


class SlotNoisePipeline:
    """Exercise generate_noise with slot-sensitive random samples, not constants."""

    def __init__(self):
        self.calls = []
        self.noise_shapes = []
        self.fail = False

    def generate_noise(self, shape, seed=None, **kwargs):
        self.noise_shapes.append(tuple(shape))
        return torch.randn(shape, generator=torch.Generator().manual_seed(seed or 0))

    def __call__(self, **kwargs):
        batch = kwargs["batch_size"]
        noise = self.generate_noise((batch, 1, 1, 1, 1), seed=kwargs["seed"])
        self.calls.append(
            {
                "batch_size": batch,
                "actions": kwargs["action"].clone(),
                "noise": noise.clone(),
            }
        )
        if self.fail:
            raise RuntimeError("synthetic pipeline failure")
        output = []
        for row in range(batch):
            memory = np.asarray(kwargs["input_image4"][row][-1]).mean()
            action = float(kwargs["action"][row, -1, 0])
            base = 70 + 12 * float(noise[row].item()) + 0.2 * memory + action * 3
            output.append(
                [
                    Image.fromarray(
                        np.full((2, 2, 3), np.uint8(np.clip(base + step, 0, 255)))
                    )
                    for step in range(kwargs["num_frames"])
                ]
            )
        return output


class SlotNoiseCosmos:
    def __init__(self):
        self.calls = []

    def query_temporal_future_batch(self, reference, memory, actions, **kwargs):
        self.calls.append(
            {
                "reference": reference.copy(),
                "memory": memory.copy(),
                "actions": actions.copy(),
                "kwargs": kwargs,
            }
        )
        rng = np.random.default_rng(kwargs["batch_seed"])
        noise = rng.integers(0, 20, size=len(actions))
        clip = np.empty((len(actions), 8, 2, 2, 3), dtype=np.uint8)
        active = kwargs.get("active_sample_mask", np.ones(len(actions), bool))
        for row in range(len(actions)):
            if active[row]:
                base = int(memory[row, -1].mean() * 0.2 + noise[row] + 60)
                clip[row] = np.stack(
                    [np.full((2, 2, 3), base + step, np.uint8) for step in range(8)]
                )
            else:
                clip[row] = memory[row, -1]
        return {"future_primary_clip": clip}


def _retry_guard(folder):
    from spatial_wovr_memory_guard import GuardSettings, MemoryGuard

    rows = iter([0, 22000, 0, 0])

    def probe(devices, run):
        return [
            {"index": 0, "free_mib": 40000, "foreign_mib": next(rows, 0), "foreign": []}
        ]

    return MemoryGuard(
        GuardSettings(enabled=True), [0], str(folder), probe=probe, sleep=lambda _: None
    )


def test_wan_oom_replay_keeps_slot_noise_and_commits_history_once(tmp_path):
    ActiveGenerationTests.setUpClass()
    case = ActiveGenerationTests("runTest")
    case.setUp()
    try:
        control, env = case.environment(compact=True), case.environment(compact=True)
        actions = np.full((4, 8, 7), 0.1, np.float32)
        expected = control._infer_next_chunk_frames(actions, active_indices=[1, 3])

        class OOMOnce(SlotNoisePipeline):
            def __call__(self, **kwargs):
                output = super().__call__(**kwargs)
                if len(self.calls) == 1:
                    raise RuntimeError(
                        "CUDA out of memory. synthetic competing process"
                    )
                return output

        env.pipe = OOMOnce()
        env.cfg.memory_guard = {"enabled": True}
        with (
            patch(
                "spatial_wovr_memory_guard.guard_for_env",
                return_value=_retry_guard(tmp_path),
            ),
            patch("torch.cuda.empty_cache"),
        ):
            actual = env._infer_next_chunk_frames(actions, active_indices=[1, 3])
        np.testing.assert_array_equal(actual, expected)
        torch.testing.assert_close(env.current_obs, control.current_obs)
        torch.testing.assert_close(env.condition_action, control.condition_action)
        assert len(env.pipe.calls) == 2
        torch.testing.assert_close(
            env.pipe.calls[0]["noise"], env.pipe.calls[1]["noise"]
        )
        for actual_history, expected_history in zip(
            env.image_queue, control.image_queue
        ):
            for actual_frame, expected_frame in zip(actual_history, expected_history):
                torch.testing.assert_close(actual_frame, expected_frame)
    finally:
        case.doCleanups()


def test_cosmos_oom_replay_keeps_request_seed_and_commits_memory_once(tmp_path):
    ActiveGenerationTests.setUpClass()
    case = ActiveGenerationTests("runTest")
    case.setUp()
    try:
        control, env = case.environment(compact=True), case.environment(compact=True)
        actions = np.full((4, 8, 7), 0.1, np.float32)
        expected = control._query_cosmos_wrist_wovr(actions, active_indices=[1, 3])

        class OOMOnce(SlotNoiseCosmos):
            def query_temporal_future_batch(self, *args, **kwargs):
                output = super().query_temporal_future_batch(*args, **kwargs)
                if len(self.calls) == 1:
                    raise RuntimeError("CUDA out of memory. synthetic service OOM")
                return output

        env.cosmos_client = OOMOnce()
        env.cfg.memory_guard = {"enabled": True}
        with (
            patch(
                "spatial_wovr_memory_guard.guard_for_env",
                return_value=_retry_guard(tmp_path),
            ),
            patch("torch.cuda.empty_cache"),
        ):
            actual = env._query_cosmos_wrist_wovr(actions, active_indices=[1, 3])
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(
            env.wrist_memory_model, control.wrist_memory_model
        )
        np.testing.assert_array_equal(
            env.wrist_condition_actions, control.wrist_condition_actions
        )
        calls = env.cosmos_client.calls
        assert len(calls) == 3  # retry first request, then process the second batch
        assert calls[0]["kwargs"]["batch_seed"] == calls[1]["kwargs"]["batch_seed"]
        np.testing.assert_array_equal(calls[0]["memory"], calls[1]["memory"])
        np.testing.assert_array_equal(calls[0]["actions"], calls[1]["actions"])
    finally:
        case.doCleanups()


class ActiveGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.env_class = production_environment_class()
        cls.mask_class = load_file(
            "actual_first_success", WORLD / "first_success.py"
        ).FirstSuccessMask
        cls.noise_module = load_file(
            "actual_generation_batch", WORLD / "generation_batch.py"
        )

    def setUp(self):
        self.module_patch = patch.dict(
            sys.modules, {"rlinf.envs.world_model.generation_batch": self.noise_module}
        )
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)

    def environment(self, *, compact=False, fast=False):
        env = object.__new__(self.env_class)
        env.num_envs = 4
        env.device = torch.device("cpu")
        env.cfg = OmegaConf.create(
            {
                "cosmos_image_size": [2, 2],
                "max_episode_steps": 240,
                "reward_coef": 5,
                "task_suite_name": "libero_spatial",
            }
        )
        env.wan_chunk = env.chunk = 8
        env.condition_frame_length = 5
        env.num_frames = 13
        env.num_inference_steps = 5
        env.retain_action = True
        env.debug_io = False
        env.fast_frame_transfer = fast
        env.skip_successful_env_generation = compact
        env._wan_reference_images = {}
        env.condition_action = torch.zeros(4, 5, 7)
        env.image_queue = [
            [torch.full((3, 1, 2, 2), -0.8 + row * 0.1) for _ in range(5)]
            for row in range(4)
        ]
        env.current_obs = torch.stack(
            [torch.full((3, 1, 13, 2, 2), -0.8 + row * 0.1) for row in range(4)]
        )
        env.pipe = SlotNoisePipeline()
        env.cosmos_client = SlotNoiseCosmos()
        env.first_success_mask = self.mask_class(4, "cpu", threshold=0.5)
        env.use_cosmos_wrist = env.cosmos_wrist_use_wovr_contract = True
        env.use_libero_state_bridge = False
        env.cosmos_wrist_cross_view_primary_history = False
        env.cosmos_wrist_primary_reference = True
        env.cosmos_gripper_action_mode = "already_cosmos"
        env.cosmos_actions_are_normalized = False
        env.cosmos_future_denoising_steps = 5
        env.cosmos_batch_size = 2
        env.wrist_reference_model = np.full((4, 2, 2, 3), 20, np.uint8)
        env.wrist_memory_model = np.stack(
            [np.full((4, 2, 2, 3), 25 + row, np.uint8) for row in range(4)]
        )
        env.wrist_condition_actions = np.zeros((4, 5, 7), np.float32)
        env.wrist_images = env.wrist_memory_model[:, -1].copy()
        env._cosmos_model_to_runtime = lambda clip: clip.copy()
        env._resize_uint8_images = lambda images, size: images.copy()
        env._update_primary_cross_view_memory = lambda clip: None
        env.image_size = (2, 2)
        env.cosmos_policy_states = torch.zeros(4, 8)
        env.task_descriptions = [f"task {row}" for row in range(4)]
        env.seed = 19
        env._elapsed_steps = 0
        env.prev_step_reward = torch.zeros(4)
        env.returns = torch.zeros(4)
        env.success_once = torch.zeros(4, dtype=torch.bool)
        env.record_metrics = True
        env.use_rel_reward = True
        env.auto_reset = False
        env.shadow_episode_info = {}
        env._libero_global_image_batch = None
        env._hybrid_generation_executor = None
        env.onload = lambda: None
        env.reward_call = 0

        def probabilities():
            values = torch.full((4, 8), 0.1)
            if env.reward_call == 0:
                values[0, 1] = values[2, 7] = 0.8
            elif env.reward_call == 1:
                values[1, 2] = 0.8
            elif env.reward_call == 2:
                values[3, 0] = 0.8
            env.reward_call += 1
            return values

        env._infer_next_chunk_rewards = probabilities
        return env

    @staticmethod
    def actions():
        actions = torch.zeros(4, 8, 7)
        actions[:, :, 0] = torch.arange(4)[:, None]
        actions[:, :, -1] = -1
        return actions

    def test_live_prefix_observations_rewards_terminations_match_full_generation(self):
        full, compact = self.environment(), self.environment(compact=True)
        for _ in range(4):
            active = ~compact.first_success_mask.finished.clone()
            expected = full.chunk_step(self.actions())
            actual = compact.chunk_step(self.actions())
            for index in (1, 2, 3):
                torch.testing.assert_close(actual[index], expected[index])
            for camera in ("main_images", "wrist_images", "states"):
                torch.testing.assert_close(
                    actual[0][0][camera][active], expected[0][0][camera][active]
                )
            for camera in ("wan", "cosmos_wrist"):
                np.testing.assert_array_equal(
                    compact._pending_debug_video_views[camera][active.numpy()],
                    full._pending_debug_video_views[camera][active.numpy()],
                )
            torch.testing.assert_close(compact.returns, full.returns)
            torch.testing.assert_close(
                compact.first_success_mask.valid_steps,
                full.first_success_mask.valid_steps,
            )

    def test_partial_success_compacts_wan_but_keeps_full_cosmos_seed_and_slots(self):
        env = self.environment(compact=True)
        env.chunk_step(self.actions())
        previous_condition = env.condition_action.clone()
        env.chunk_step(self.actions())
        self.assertEqual([call["batch_size"] for call in env.pipe.calls], [4, 2])
        self.assertEqual(env.pipe.noise_shapes, [(4, 1, 1, 1, 1)] * 2)
        torch.testing.assert_close(
            env.pipe.calls[1]["noise"], env.pipe.calls[0]["noise"][[1, 3]]
        )
        torch.testing.assert_close(
            env.condition_action[[0, 2]], previous_condition[[0, 2]]
        )
        self.assertEqual(
            [len(call["actions"]) for call in env.cosmos_client.calls], [2] * 4
        )
        self.assertEqual(
            [call["kwargs"]["batch_seed"] for call in env.cosmos_client.calls],
            [19, 21, 27, 29],
        )
        for call in env.cosmos_client.calls[2:]:
            np.testing.assert_array_equal(
                call["kwargs"]["active_sample_mask"], [False, True]
            )
        self.assertNotIn("generate_noise", vars(env.pipe))

    def test_all_success_has_no_model_calls_and_freezes_rectangular_outputs(self):
        env = self.environment(compact=True)
        for _ in range(3):
            env.chunk_step(self.actions())
        self.assertTrue(env.first_success_mask.finished.all())
        before = {
            "main": env._wrap_obs()["main_images"].clone(),
            "wrist": env.wrist_images.copy(),
            "returns": env.returns.clone(),
            "valid": env.first_success_mask.valid_steps.clone(),
            "condition": env.condition_action.clone(),
            "wan_calls": len(env.pipe.calls),
            "rpc": len(env.cosmos_client.calls),
        }
        result = env.chunk_step(self.actions())
        self.assertEqual(len(env.pipe.calls), before["wan_calls"])
        self.assertEqual(len(env.cosmos_client.calls), before["rpc"])
        self.assertEqual(result[1].shape, (4, 8))
        self.assertEqual(result[2].shape, (4, 8))
        self.assertEqual(result[0][0]["main_images"].shape, (4, 2, 2, 3))
        self.assertEqual(env.current_obs.shape, (4, 3, 1, 13, 2, 2))
        self.assertEqual(torch.count_nonzero(result[1]), 0)
        self.assertFalse(result[2].any())
        torch.testing.assert_close(result[0][0]["main_images"], before["main"])
        np.testing.assert_array_equal(env.wrist_images, before["wrist"])
        torch.testing.assert_close(env.returns, before["returns"])
        torch.testing.assert_close(env.first_success_mask.valid_steps, before["valid"])
        torch.testing.assert_close(env.condition_action, before["condition"])
        self.assertEqual(env.elapsed_steps, 32)

    def test_fast_transfer_preserves_pixels_and_caches_fixed_references(self):
        baseline, fast = self.environment(), self.environment(fast=True)
        for _ in range(2):
            expected = baseline._infer_next_chunk_frames(self.actions())
            actual = fast._infer_next_chunk_frames(self.actions())
            np.testing.assert_array_equal(actual, expected)
            torch.testing.assert_close(fast.current_obs, baseline.current_obs)
        self.assertEqual(set(fast._wan_reference_images), set(range(4)))
        self.assertEqual(baseline._wan_reference_images, {})
        cached = fast._wan_reference_images[1]
        fast._infer_next_chunk_frames(self.actions())
        self.assertIs(fast._wan_reference_images[1], cached)

    def test_full_and_partial_reset_clear_exact_reference_cache_rows(self):
        env = self.environment(fast=True)
        env._infer_next_chunk_frames(self.actions())
        references = env._wan_reference_images.copy()
        env.first_success_mask.finished[:] = True
        env._reset_metrics([1, 3])
        self.assertEqual(set(env._wan_reference_images), {0, 2})
        self.assertIs(env._wan_reference_images[0], references[0])
        torch.testing.assert_close(
            env.first_success_mask.finished, torch.tensor([True, False, True, False])
        )
        env._infer_next_chunk_frames(self.actions())
        env.first_success_mask.finished[:] = True
        env._reset_metrics(torch.tensor(2))
        self.assertEqual(set(env._wan_reference_images), {0, 1, 3})
        torch.testing.assert_close(
            env.first_success_mask.finished, torch.tensor([True, True, False, True])
        )
        env._infer_next_chunk_frames(self.actions())
        env.first_success_mask.finished[:] = True
        env._reset_metrics(torch.tensor([True, False, False, True]))
        self.assertEqual(set(env._wan_reference_images), {1, 2})
        torch.testing.assert_close(
            env.first_success_mask.finished, torch.tensor([False, True, True, False])
        )
        env._reset_metrics()
        self.assertEqual(env._wan_reference_images, {})
        self.assertFalse(env.first_success_mask.finished.any())

    def test_generation_exception_restores_pipeline_noise_method(self):
        env = self.environment(compact=True)
        original = env.pipe.generate_noise
        env.pipe.fail = True
        with self.assertRaisesRegex(RuntimeError, "synthetic pipeline failure"):
            env._infer_next_chunk_frames(self.actions(), active_indices=[1, 3])
        self.assertNotIn("generate_noise", vars(env.pipe))
        self.assertEqual(env.pipe.generate_noise, original)
        self.assertEqual(env.pipe.noise_shapes, [(4, 1, 1, 1, 1)])

    def test_native_grpo_gradients_match_and_terminal_suffix_has_zero_gradient(self):
        required = [
            NATIVE / "utils/metric_utils.py",
            NATIVE / "utils/utils.py",
            NATIVE / "algorithms/advantages.py",
            NATIVE / "algorithms/losses.py",
        ]
        if not all(path.is_file() for path in required):
            self.skipTest("Native actor loss sources are not installed")
        namespace = {"torch": torch, "Optional": Optional, "Callable": Callable}
        extract_functions(required[0], {"compute_loss_mask"}, namespace)
        extract_functions(required[1], {"masked_mean", "masked_mean_ratio"}, namespace)
        extract_functions(required[2], {"compute_grpo_advantages"}, namespace)
        extract_functions(required[3], {"compute_ppo_actor_loss"}, namespace)
        gradients, masks = [], []
        for env in (self.environment(), self.environment(compact=True)):
            rewards, dones = [], [torch.zeros(4, 8, dtype=torch.bool)]
            for _ in range(2):
                _, reward, terminal, truncated, _ = env.chunk_step(self.actions())
                rewards.append(reward)
                dones.append(terminal | truncated)
            loss_mask, mask_sum = namespace["compute_loss_mask"](torch.stack(dones))
            scores = (torch.stack(rewards) * loss_mask).sum(dim=(0, 2))
            mask = loss_mask.transpose(1, 2).reshape(16, 4)
            lengths = mask_sum.transpose(1, 2).reshape(16, 4)
            advantages, _ = namespace["compute_grpo_advantages"](
                scores, mask, group_size=4
            )
            logprobs = torch.zeros(16, 4, 7, requires_grad=True)
            loss, _ = namespace["compute_ppo_actor_loss"](
                logprobs=logprobs,
                old_logprobs=torch.zeros_like(logprobs),
                advantages=advantages[..., None],
                loss_mask=mask[..., None],
                loss_mask_sum=lengths[..., None],
                max_episode_steps=16,
                clip_ratio_low=0.2,
                clip_ratio_high=0.28,
            )
            loss.backward()
            self.assertTrue(torch.isfinite(logprobs.grad).all())
            self.assertGreater(torch.count_nonzero(logprobs.grad[mask]), 0)
            self.assertEqual(torch.count_nonzero(logprobs.grad[~mask]), 0)
            gradients.append(logprobs.grad)
            masks.append(mask)
        torch.testing.assert_close(gradients[0], gradients[1])
        torch.testing.assert_close(masks[0], masks[1])
        torch.testing.assert_close(masks[1].sum(dim=0), torch.tensor([2, 11, 8, 16]))


if __name__ == "__main__":
    unittest.main()
