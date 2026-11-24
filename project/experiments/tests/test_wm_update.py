"""Truth alignment, complete resume, and pair acceptance without GPU allocation."""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "third_party/WorldPilot"))

from cosmos_bridge.checkpoint_identity import checkpoint_tree_identity

from wm_update import (
    collection_environment,
    gpu_ids,
    gpu_pool,
    training_environment,
    training_plan,
    worker_command,
)
from wm_update_accept import compare, evaluation_manifest, package_pair
from wm_update_collect import (
    collect_episode,
    collection_plan,
    merge_collection,
)
from wm_update_data import (
    CosmosUpdateDataset,
    PairedWindowDataset,
    WanUpdateDataset,
    atomic_json,
    build_manifest,
    inventory,
    load_manifest,
    read_window,
    sha256,
    window_indices,
    write_episode,
)
from wm_update_wan import fixed_rng, load_state, save_state


class TruthTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def episode(
        self, *, number=0, task=0, role="new", initial=None, count=32, success=False
    ):
        frames = [np.full((16, 16, 3), i * 3, np.uint8) for i in range(count + 1)]
        actions = np.zeros((count, 7), np.float32)
        actions[:, 0] = np.arange(count) / 100
        actions[:, -1] = np.where(np.arange(count) >= 16, 1, -1)
        metadata = {
            "episode_id": f"episode-{task}-{number}",
            "task_id": task,
            "task": f"task {task}",
            "suite": "libero_spatial",
            "initial_state_id": number if initial is None else initial,
            "policy_id": "policy-10",
            "seed": number,
            "success": success,
        }
        labels = np.zeros(count, bool)
        labels[-1] = success
        path = self.root / f"episode-{task}-{number}.hdf5"
        write_episode(
            path,
            primary=frames,
            wrist=frames,
            actions=actions,
            states=np.zeros((count + 1, 5)),
            proprio=np.zeros((count + 1, 9)),
            successes=labels,
            rewards=labels.astype(float),
            dones=labels,
            metadata=metadata,
        )
        return inventory(path, role=role, verify_rgb=True)

    def manifest(self):
        rows = [
            self.episode(
                number=n, task=t, role="new" if n % 2 else "old", success=n % 2 == 0
            )
            for t in range(2)
            for n in range(4)
        ]
        path = self.root / "manifest.json"
        build_manifest(rows, path)
        return path

    def test_bootstrap_window_has_five_real_initial_observations(self):
        rgb, actions = window_indices(0, 32)
        self.assertEqual(rgb.tolist(), [0] * 5 + list(range(1, 9)))
        self.assertEqual(actions.tolist(), [-1] * 5 + list(range(8)))

    def test_nonzero_window_uses_incoming_history_actions(self):
        row = self.episode()
        sample = read_window(row, 8)
        self.assertEqual(
            sample["frame_indices"].tolist(), [0, 5, 6, 7, 8, *range(9, 17)]
        )
        self.assertEqual(
            sample["action_indices"].tolist(), [-1, 4, 5, 6, 7, *range(8, 16)]
        )
        np.testing.assert_allclose(sample["actions"][1:, 0], np.arange(4, 16) / 100)
        self.assertEqual(sample["primary"].shape, (13, 16, 16, 3))

    def test_future_tail_is_never_padded(self):
        window_indices(24, 32)
        with self.assertRaises(ValueError):
            window_indices(25, 32)

    def test_atomic_writer_rejects_missing_successor(self):
        row = self.episode()
        with h5py.File(row["path"]) as file:
            metadata = json.loads(file.attrs["metadata_json"])
        path = self.root / "bad.hdf5"
        with self.assertRaises(ValueError):
            write_episode(
                path,
                primary=[np.zeros((2, 2, 3), np.uint8)] * 8,
                wrist=[],
                actions=np.ones((8, 7)),
                states=[],
                proprio=[],
                successes=[],
                rewards=[],
                dones=[],
                metadata=metadata,
            )
        self.assertFalse(path.exists())

    def test_invalid_signed_gripper_is_rejected(self):
        row = self.episode()
        with h5py.File(row["path"], "r+") as file:
            file["actions"][2, -1] = 0.5
        with self.assertRaises(ValueError):
            inventory(Path(row["path"]))

    def test_success_is_simulator_truth(self):
        row = self.episode()
        with h5py.File(row["path"], "r+") as file:
            file["successes"][-1] = True
        with self.assertRaises(ValueError):
            inventory(Path(row["path"]))

    def test_known_initial_state_group_never_crosses_split(self):
        rows = [self.episode(number=n, initial=n // 2) for n in range(8)]
        manifest = build_manifest(rows, self.root / "groups.json")
        for initial in range(4):
            self.assertEqual(
                len(
                    {
                        r["split"]
                        for r in manifest["episodes"]
                        if r["initial_state_id"] == initial
                    }
                ),
                1,
            )

    def test_duplicate_trajectory_rejected(self):
        row = self.episode()
        with self.assertRaises(ValueError):
            build_manifest([row, row], self.root / "duplicate.json")

    def test_manifest_and_source_are_immutable(self):
        path = self.manifest()
        manifest = load_manifest(path)
        row = manifest["episodes"][0]
        with h5py.File(row["path"], "r+") as file:
            file.attrs["changed"] = True
        with self.assertRaises(ValueError):
            read_window(row, 0)
        manifest["seed"] += 1
        atomic_json(path, manifest)
        with self.assertRaises(ValueError):
            load_manifest(path)

    def test_sampling_is_reproducible_across_restart(self):
        path = self.manifest()
        first = PairedWindowDataset(str(path))
        resumed = PairedWindowDataset(str(path))
        for index in [0, 1, 19, 1999]:
            a, b = first[index], resumed[index]
            self.assertEqual(
                (a["episode"]["trajectory_id"], a["step"]),
                (b["episode"]["trajectory_id"], b["step"]),
            )

    def test_wan_dataset_contract(self):
        sample = WanUpdateDataset(str(self.manifest()))[0]
        self.assertEqual(len(sample["video"]), 13)
        self.assertEqual(sample["video"][0].size, (256, 256))
        self.assertEqual(sample["action"].shape, (13, 7))

    def test_cosmos_dataset_matches_inference_preprocessing(self):
        path = self.manifest()
        stats = self.root / "stats.json"
        atomic_json(stats, {"actions_min": [-1] * 7, "actions_max": [1] * 7})
        data = CosmosUpdateDataset(str(path), stats_path=str(stats))
        sample = data[0]
        self.assertEqual(sample["video"].shape, (3, 13, 224, 224))
        self.assertEqual(sample["video"].dtype, torch.uint8)
        self.assertEqual(sample["action"].shape, (13, 7))
        np.testing.assert_array_equal(sample["action"][0], [0, 0, 0, 0, 0, 0, -1])
        self.assertEqual(sample["num_conditional_frames"], 2)

    def test_evaluation_uses_only_held_out_episodes(self):
        path = self.manifest()
        result = evaluation_manifest(path, self.root / "eval.json")
        self.assertTrue(all(row["split"] == "val" for row in result["episodes"]))
        self.assertEqual(
            result["update_manifest_sha256"], load_manifest(path)["manifest_sha256"]
        )

    def test_collection_plan_requires_exact_checkpoint(self):
        config = self.root / "policy.yaml"
        config.write_text("rollout: {}\n")
        checkpoint = self.root / "actor/model_state_dict/full_weights.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"fixed policy")
        plan = collection_plan(
            policy_config=config,
            policy_checkpoint=self.root,
            per_task=2,
            seed=1,
            max_steps=240,
            settle_steps=10,
        )
        self.assertEqual(plan["policy_sha256"], sha256(checkpoint))
        self.assertEqual(plan["execute_actions"], 8)

    def test_incomplete_collection_cannot_be_merged(self):
        atomic_json(self.root / "collection_plan.json", {"per_task": 2})
        with self.assertRaises(FileNotFoundError):
            merge_collection(self.root, 4)

    def test_pair_package_rejects_different_training_data(self):
        path = self.manifest()
        wan, cosmos = self.root / "wan", self.root / "iter_000000200"
        atomic_json(wan / "wm_update.json", {"manifest_sha256": "wrong", "step": 200})
        atomic_json(
            cosmos / "wm_update_complete.json",
            {"identity": {"manifest": "wrong"}, "step": 200},
        )
        with self.assertRaises(ValueError):
            package_pair(
                types.SimpleNamespace(
                    manifest=path,
                    wan_checkpoint=wan,
                    cosmos_checkpoint=cosmos,
                    output=self.root / "pair",
                )
            )

    def test_pair_package_preserves_sources_and_records_shard_checksums(self):
        path = self.manifest()
        identity = load_manifest(path)["manifest_sha256"]
        wan, cosmos = self.root / "wan", self.root / "iter_000000200"
        atomic_json(wan / "wm_update.json", {"manifest_sha256": identity, "step": 200})
        (wan / "model-00001.safetensors").write_bytes(b"test-weights")
        (wan / "Wan2.2_VAE.pth").write_bytes(b"test-vae")
        atomic_json(
            cosmos / "wm_update_complete.json",
            {"identity": {"manifest": identity}, "step": 200},
        )
        (cosmos / "model").mkdir()
        (cosmos / "model/.metadata").write_bytes(b"test-metadata")
        (cosmos / "model/__0_0.distcp").write_bytes(b"test-shard")
        with contextlib.redirect_stdout(io.StringIO()):
            result = package_pair(
                types.SimpleNamespace(
                    manifest=path,
                    wan_checkpoint=wan,
                    cosmos_checkpoint=cosmos,
                    output=self.root / "pair",
                )
            )
        self.assertEqual(Path(result["wan"]).resolve(), wan)
        files = json.loads(
            (self.root / "pair/cosmos/checkpoint_manifest.json").read_text()
        )["files"]
        for row in files:
            self.assertEqual(
                sha256(self.root / "pair/cosmos" / row["path"]), row["sha256"]
            )
        identity = checkpoint_tree_identity(result["cosmos"])
        self.assertEqual(identity["file_count"], len(files))
        self.assertEqual(
            {row["relative_path"] for row in identity["files"]},
            {"model/.metadata", "model/__0_0.distcp"},
        )


class CollectorTests(unittest.TestCase):
    def test_only_executed_actions_and_successors_are_saved(self):
        fake = types.ModuleType("experiments.audit_rlinf_pi05_wan_libero_pair")
        fake.rot180 = lambda value: value
        fake.proprio_from_obs = lambda obs: np.zeros(9)
        fake.policy_observation = lambda obs, task: {"states": torch.ones(1, 8)}
        fake.settle_env = lambda env, initial, settle: env.observation()

        class Env:
            step_id = 0

            @property
            def sim(self):
                return types.SimpleNamespace(get_state=lambda: np.array([self.step_id]))

            def observation(self):
                rgb = np.full((2, 2, 3), self.step_id, np.uint8)
                return {"agentview_image": rgb, "robot0_eye_in_hand_image": rgb}

            def step(self, action):
                self.step_id += 1
                return self.observation(), 0, False, {}

            def check_success(self):
                return self.step_id == 9

        class Model:
            calls = 0

            def predict_action_batch(self, *, env_obs, **kwargs):
                assert not env_obs["states"].any()
                self.calls += 1
                value = torch.zeros(1, 10, 7)
                value[:, 8:] = 99  # These two predicted actions must never execute.
                return value, {}

        model = Model()
        with patch.dict(sys.modules, {fake.__name__: fake}):
            result = collect_episode(
                Env(),
                model,
                initial_state=np.zeros(1),
                task="test",
                max_steps=240,
                settle_steps=10,
                mode="train",
            )
        self.assertEqual(len(result["actions"]), 9)
        self.assertEqual(len(result["primary"]), 10)
        self.assertEqual(model.calls, 2)
        self.assertEqual(result["primary"][-1][0, 0, 0], 9)
        self.assertFalse(np.any(np.asarray(result["actions"]) == 99))

    def test_native_primary_capture_is_opt_in_and_preserves_metric_shapes(self):
        import wm_eval_predict

        class FakePredictor:
            def __init__(self, steps):
                pass

            def reset(self, *args):
                pass

            def chunk(self, actions, seed):
                return np.zeros((8, 256, 256, 3), np.uint8)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {
                "trajectory_id": "episode",
                "task_id": 0,
                "task": "task",
                "horizon": 16,
            }
            atomic_json(
                root / "manifest.json",
                {"episodes": [row], "manifest_sha256": "fixed", "protocol": "test"},
            )
            rgb = np.zeros((17, 256, 256, 3), np.uint8)
            actions = np.zeros((16, 7), np.float32)
            for capture in ["0", "1"]:
                args = types.SimpleNamespace(
                    model="wan",
                    manifest=root / "manifest.json",
                    output=root / capture,
                    shard_index=0,
                    num_shards=1,
                    steps=5,
                    seed=42,
                    warmup=1,
                )
                with (
                    patch.dict(os.environ, WM_EVAL_CAPTURE_NATIVE_PRIMARY=capture),
                    patch.object(wm_eval_predict, "WanPredictor", FakePredictor),
                    patch.object(
                        wm_eval_predict,
                        "read_episode",
                        return_value=(rgb, rgb, actions),
                    ),
                    patch.object(
                        wm_eval_predict, "checkpoint_identity", return_value=[]
                    ),
                    patch.object(torch.cuda, "is_available", return_value=True),
                    patch.object(
                        torch.cuda, "get_device_name", return_value="CPU fixture"
                    ),
                    patch.object(torch.cuda, "synchronize"),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    wm_eval_predict.predict(args)
                with np.load(args.output / "wan/episode.npz") as data:
                    self.assertEqual(data["prediction"].shape, (16, 224, 224, 3))
                    self.assertEqual("native_prediction" in data, capture == "1")
                    if capture == "1":
                        self.assertEqual(
                            data["native_prediction"].shape, (16, 256, 256, 3)
                        )
            self.assertFalse(torch.cuda.is_initialized())


class ResumeTests(unittest.TestCase):
    def test_complete_state_restores_optimizer_and_next_update(self):
        # A real CPU backward/save/reload/next-backward comparison exercises
        # the same DCP state helpers used by the four-GPU trainer.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            torch.manual_seed(7)
            model = torch.nn.Linear(3, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            inputs = torch.tensor([[1.0, 2, 3]])

            def update(module, opt):
                opt.zero_grad()
                module(inputs).square().mean().backward()
                opt.step()

            update(model, optimizer)
            save_state(path, model, optimizer, step=1, identity={"manifest": "fixed"})
            update(model, optimizer)
            expected = {key: value.clone() for key, value in model.state_dict().items()}
            restored = torch.nn.Linear(3, 1)
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
            step = load_state(
                path, restored, restored_optimizer, identity={"manifest": "fixed"}
            )
            self.assertEqual(step, 1)
            update(restored, restored_optimizer)
            for key, value in restored.state_dict().items():
                torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
            with self.assertRaises(ValueError):
                load_state(
                    path, restored, restored_optimizer, identity={"manifest": "changed"}
                )

    def test_validation_does_not_change_training_rng(self):
        torch.manual_seed(7)
        state = torch.get_rng_state().clone()
        with fixed_rng(42):
            torch.randn(3)
        torch.testing.assert_close(torch.get_rng_state(), state)


class CommandTests(unittest.TestCase):
    def test_collection_render_and_policy_share_the_physical_gpu(self):
        env = collection_environment("5")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "5")
        self.assertEqual(env["MUJOCO_EGL_DEVICE_ID"], "5")

    def test_cosmos_sampler_continues_exact_rank_indices(self):
        from wm_update_cosmos import UpdateSampler

        for rank in range(4):
            full = list(UpdateSampler(0, 5, 4, rank, 4))
            sampler = UpdateSampler(2, 5, 4, rank, 4)
            sampler.set_epoch(0)
            self.assertEqual(list(sampler), full[8:])

    def test_validation_sample_spans_all_tasks(self):
        from wm_update_data import validation_indices

        indices = [i for rank in range(4) for i in validation_indices(100, rank, 4)]
        self.assertEqual(len(set(indices)), 16)
        self.assertEqual((min(indices), max(indices)), (0, 99))

    def test_false_success_is_measured_against_simulator_truth(self):
        from wm_update_accept import summarize_outcomes

        report = summarize_outcomes(
            [
                {"success": False, "prediction_max": 0.95, "truth_max": 0.3},
                {"success": True, "prediction_max": 0.96, "truth_max": 0.97},
            ]
        )
        self.assertEqual(report["predicted_video"]["fp"], 1)
        self.assertEqual(report["real_video"]["fp"], 0)
        self.assertEqual(report["classifier_disagreements"], 1)

    def test_gpu_ids_are_explicit_and_distinct(self):
        self.assertEqual(gpu_ids("0,1,2,3"), ["0", "1", "2", "3"])
        for value in ["0,0,1,2", "0,1", "all", "0,1,2,-1"]:
            with self.assertRaises(ValueError):
                gpu_ids(value)

    def test_training_uses_isolated_two_rank_torchrun(self):
        args = types.SimpleNamespace(
            manifest=Path("manifest.json"),
            output=Path("output"),
            settings=Path("settings.json"),
            steps=2000,
            save_interval=200,
            accumulation=8,
            lr=1e-6,
            seed=42,
            resume=None,
        )
        with patch.dict(
            os.environ, RLINF_ENV="/runtime", COSMOS_PYTHON="/cosmos/python"
        ):
            command = worker_command(args, "wan")
        self.assertIn("--standalone", command)
        self.assertIn("--nproc_per_node=2", command)

    def training_args(self, model="both", gpus="0,1,2,3"):
        return types.SimpleNamespace(
            manifest=Path("manifest.json"),
            output=Path("output"),
            settings=Path("settings.json"),
            steps=2000,
            save_interval=200,
            accumulation=8,
            lr=1e-6,
            seed=42,
            resume=None,
            model=model,
            gpus=gpus,
        )

    def test_parallel_models_use_disjoint_pairs_and_keep_batch_16(self):
        with patch.dict(
            os.environ, RLINF_ENV="/runtime", COSMOS_PYTHON="/cosmos/python"
        ):
            jobs = training_plan(self.training_args(gpus="4,5,6,7"))
        self.assertEqual(
            [(j["model"], j["gpus"]) for j in jobs],
            [("wan", ["4", "5"]), ("cosmos", ["6", "7"])],
        )
        self.assertTrue(all(j["effective_batch"] == 16 for j in jobs))
        self.assertTrue(all("--nproc_per_node=2" in j["command"] for j in jobs))
        self.assertNotEqual(jobs[0]["log"], jobs[1]["log"])

    def test_single_model_resume_keeps_its_pair_from_the_four_gpu_pool(self):
        with patch.dict(
            os.environ, RLINF_ENV="/runtime", COSMOS_PYTHON="/cosmos/python"
        ):
            jobs = training_plan(self.training_args(model="cosmos"))
        self.assertEqual(jobs[0]["gpus"], ["2", "3"])
        self.assertEqual(len(jobs), 1)

    def test_single_model_can_select_only_its_two_gpus(self):
        with patch.dict(
            os.environ, RLINF_ENV="/runtime", COSMOS_PYTHON="/cosmos/python"
        ):
            jobs = training_plan(self.training_args(model="wan", gpus="5,7"))
        self.assertEqual(jobs[0]["gpus"], ["5", "7"])
        with self.assertRaises(ValueError):
            training_plan(self.training_args(gpus="5,7"))

    def test_eight_gpu_pool_uses_four_ranks_per_model_and_batch_16(self):
        args = self.training_args(gpus="0,1,2,3,4,5,6,7")
        args.accumulation = 4
        with patch.dict(
            os.environ, RLINF_ENV="/runtime", COSMOS_PYTHON="/cosmos/python"
        ):
            jobs = training_plan(args)
        self.assertEqual(
            [j["gpus"] for j in jobs], [["0", "1", "2", "3"], ["4", "5", "6", "7"]]
        )
        self.assertTrue(
            all(j["num_processes"] == 4 and j["effective_batch"] == 16 for j in jobs)
        )
        self.assertTrue(all("--nproc_per_node=4" in j["command"] for j in jobs))

    def test_four_card_single_model_resume_requires_explicit_group_size(self):
        args = self.training_args(model="cosmos", gpus="4,5,6,7")
        args.ranks_per_model = 4
        with patch.dict(
            os.environ, RLINF_ENV="/runtime", COSMOS_PYTHON="/cosmos/python"
        ):
            jobs = training_plan(args)
        self.assertEqual(jobs[0]["gpus"], ["4", "5", "6", "7"])
        self.assertEqual(jobs[0]["num_processes"], 4)
        with self.assertRaises(ValueError):
            training_plan(self.training_args(gpus="0,1,2,3,4,5"))

    def test_collection_accepts_four_or_eight_unique_cards(self):
        self.assertEqual(len(gpu_pool("0,1,2,3,4,5,6,7")), 8)
        for value in ("0,1", "0,1,2,3,4,5", "0,1,2,3,4,5,6,6"):
            with self.assertRaises(ValueError):
                gpu_pool(value)

    def test_eight_collectors_have_disjoint_shards_and_merge_all_eight(self):
        from wm_update import main

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.yaml"
            config.write_text("rollout: {}\n")
            output = root / "collection"
            command = [
                "wm_update.py",
                "collect",
                "--gpus",
                "0,1,2,3,4,5,6,7",
                "--policy-config",
                str(config),
                "--policy-checkpoint",
                str(root),
                "--output",
                str(output),
            ]
            with (
                patch.object(sys, "argv", command),
                patch("wm_update.environment"),
                patch("wm_update.require_idle"),
                patch("wm_update.runtime"),
                patch(
                    "wm_update_collect.collection_plan",
                    return_value={"suite": "libero_spatial"},
                ),
                patch("wm_update_collect.policy_config"),
                patch("wm_eval.supervise") as supervise,
                patch(
                    "wm_update_collect.merge_collection", return_value={"summary": {}}
                ) as merge,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(main(), 0)
            commands = supervise.call_args.args[0]
            self.assertEqual(len(commands), 8)
            self.assertEqual(
                {env["CUDA_VISIBLE_DEVICES"] for _, env, _ in commands},
                set(map(str, range(8))),
            )
            self.assertEqual(
                {cmd[cmd.index("--shard-index") + 1] for cmd, _, _ in commands},
                set(map(str, range(8))),
            )
            self.assertTrue(
                all(cmd[cmd.index("--num-shards") + 1] == "8" for cmd, _, _ in commands)
            )
            merge.assert_called_once_with(output, 8)

    def test_training_environment_drops_inherited_distributed_rank(self):
        with patch.dict(
            os.environ, RANK="7", MASTER_PORT="1234", TORCHELASTIC_RUN_ID="old"
        ):
            first, second = (
                training_environment(["0", "1"]),
                training_environment(["2", "3"]),
            )
        self.assertEqual(first["CUDA_VISIBLE_DEVICES"], "0,1")
        self.assertEqual(second["CUDA_VISIBLE_DEVICES"], "2,3")
        for key in ["RANK", "MASTER_PORT", "TORCHELASTIC_RUN_ID"]:
            self.assertNotIn(key, first)
            self.assertNotIn(key, second)

    def test_comparison_refuses_different_manifests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {"model": "wan", "manifest_sha256": "a"}
            atomic_json(
                root / "baseline/metrics.json", [row, dict(row, model="cosmos")]
            )
            atomic_json(
                root / "candidate/metrics.json",
                [dict(row, manifest_sha256="b"), dict(row, model="cosmos")],
            )
            with (
                self.assertRaises(ValueError),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                compare(
                    types.SimpleNamespace(
                        baseline=root / "baseline",
                        candidate=root / "candidate",
                        output=root / "comparison",
                        max_regression=0.05,
                    )
                )


if __name__ == "__main__":
    unittest.main()
