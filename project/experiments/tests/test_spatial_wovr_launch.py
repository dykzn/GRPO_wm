"""CPU regression checks for dual-view eight-GPU orchestration."""

import asyncio
import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from omegaconf import OmegaConf

EXPERIMENTS = Path(__file__).resolve().parents[1]
PROJECT = EXPERIMENTS.parent
sys.path.insert(0, str(EXPERIMENTS))

from spatial_wovr_launch import (
    link_sources,
    make_endpoints,
    metadata_payload,
    prepare_runtime,
    stop_process,
    validate_batch,
    validate_server_metadata,
    validate_wrist_runtime,
    wrist_records,
)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


endpoint_module = load_module(
    "cosmos_endpoints",
    PROJECT / "third_party/RLinf/rlinf/envs/world_model/cosmos_endpoints.py",
)


class DeploymentTests(unittest.TestCase):
    def test_services_follow_environment_rank_with_reordered_visible_gpus(self):
        endpoints = make_endpoints("7,3,5,1,6,2,4,0", 18059)
        for rank in range(8):
            info = SimpleNamespace(rank=rank, group_world_size=8)
            self.assertEqual(
                endpoint_module.resolve_cosmos_endpoint(
                    {"cosmos_endpoints": endpoints}, info
                ),
                ("127.0.0.1", 18059 + rank),
            )
        with self.assertRaises(ValueError):
            make_endpoints("0,1,2,3,4,5,6,6", 18059)
        with self.assertRaises(ValueError):
            endpoint_module.resolve_cosmos_endpoint(
                {"cosmos_endpoints": endpoints},
                SimpleNamespace(rank=0, group_world_size=1),
            )

    def test_single_server_configs_remain_compatible(self):
        self.assertEqual(
            endpoint_module.resolve_cosmos_endpoint({"cosmos_port": 8059}),
            ("127.0.0.1", 8059),
        )

    def test_source_overlay_keeps_native_backend_and_modified_wrist(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base, modified = root / "base", root / "modified"
            for source in [base, modified]:
                (source / "package").mkdir(parents=True)
            (base / "package/loader.py").write_text("native loader")
            (base / "package/wrist.py").write_text("legacy wrist")
            (modified / "package/wrist.py").write_text("v3 wrist")
            target = root / "runtime"
            link_sources(target, [base, modified])
            self.assertEqual(
                (target / "package/loader.py").read_text(), "native loader"
            )
            self.assertEqual((target / "package/wrist.py").read_text(), "v3 wrist")
            self.assertEqual((base / "package/wrist.py").read_text(), "legacy wrist")

    def test_launcher_builds_and_records_the_offload_barrier_without_editing_native(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline, project, cosmos = (
                root / "baseline",
                root / "project",
                root / "cosmos",
            )
            native = Path(
                "/data41/qianb2608/wan/RLinf_opd/rlinf/runners/embodied_runner.py"
            )
            native_video = native.parents[1] / "envs/wrappers/record_video.py"
            if not native.is_file() or not native_video.is_file():
                self.skipTest("Installed runner/video source is not present")
            fixtures = {
                baseline / "rlinf/workers/env/env_worker.py": "",
                baseline / "rlinf/runners/embodied_runner.py": native.read_text(),
                baseline / "rlinf/envs/world_model/__init__.py": "",
                baseline / "rlinf/envs/wrappers/__init__.py": "",
                baseline
                / "rlinf/envs/wrappers/record_video.py": native_video.read_text(),
                baseline / "rlinf/scheduler.py": "native scheduler",
                cosmos / "cosmos_policy/_src/predict2/datasets/utils.py": "",
                cosmos / "cosmos_policy/datasets/libero_dataset.py": "",
                project / "third_party/RLinf/rlinf/envs/world_model/__init__.py": "",
                project / "third_party/RLinf/rlinf/envs/__init__.py": "",
                project / "third_party/RLinf/rlinf/envs/action_utils.py": "",
            }
            for path, source in fixtures.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(source)
            benchmark = root / "benchmark"
            for suffix in (
                "assets",
                "bddl_files/libero_spatial",
                "init_files/libero_spatial",
            ):
                (benchmark / suffix).mkdir(parents=True)
            environment = {
                "PROJECT_ROOT": str(project),
                "RLINF_BASE_ROOT": str(baseline),
                "COSMOS_BASE_ROOT": str(cosmos),
                "COSMOS_WRIST_CHECKPOINT": str(root / "checkpoint"),
                "WORLDPILOT_ROOT": str(root / "worldpilot"),
                "LIBERO_ROOT": str(root / "libero"),
                "OPENPI_ROOT": str(root / "openpi"),
                "DIFFSYNTH_ROOT": str(root / "diffsynth"),
                "LIBERO_BENCHMARK_ROOT": str(benchmark),
                "LIBERO_CONFIG_PATH": str(root / "libero_config"),
            }
            run = root / "run"
            with (
                patch.dict("os.environ", environment),
                patch.object(sys, "path", list(sys.path)),
            ):
                prepare_runtime(run)
            metadata = json.loads((run / "source_paths.json").read_text())
            overlay = Path(metadata["runner_overlay"])
            self.assertFalse(overlay.is_symlink())
            self.assertIn("env_handle.wait()", overlay.read_text())
            self.assertEqual(
                (baseline / "rlinf/runners/embodied_runner.py").read_text(),
                native.read_text(),
            )
            self.assertTrue(metadata["wait_for_env_offload_before_update"])
            video_overlay = Path(metadata["record_video_overlay"])
            self.assertFalse(video_overlay.is_symlink())
            self.assertFalse(video_overlay.parent.is_symlink())
            self.assertIn("SampledVideoMixin", video_overlay.read_text())
            self.assertEqual(
                (baseline / "rlinf/envs/wrappers/record_video.py").read_text(),
                native_video.read_text(),
            )
            import hashlib

            self.assertEqual(
                metadata["runner_overlay_sha256"],
                hashlib.sha256(overlay.read_bytes()).hexdigest(),
            )
            self.assertTrue((run / "code/rlinf/rlinf/scheduler.py").is_symlink())

    def test_nominal_rollout_is_divisible_and_old_64_env_count_is_rejected(self):
        cfg = OmegaConf.load(EXPERIMENTS / "config/spatial_wovr_pi05_8gpu.yaml")
        self.assertEqual(validate_batch(cfg), 30720)
        cfg.env.train.total_num_envs = 64
        with self.assertRaisesRegex(ValueError, "15360"):
            validate_batch(cfg)

    def test_world_model_handoff_rejects_pipeline_and_next_rollout_bootstrap_overlap(
        self,
    ):
        for field in ("use_training_pipeline", "overlap_env_bootstrap"):
            cfg = OmegaConf.load(EXPERIMENTS / "config/spatial_wovr_pi05_8gpu.yaml")
            cfg.runner[field] = True
            with self.assertRaisesRegex(ValueError, "World-model memory handoff"):
                validate_batch(cfg)

    def test_wrong_horizon_and_double_gripper_conversion_are_rejected(self):
        for field, value in [
            ("actor.model.openpi.action_horizon", 8),
            ("env.train.cosmos_gripper_action_mode", "libero_01_to_cosmos"),
            ("env.train.cosmos_wrist_cross_view_primary_history", True),
        ]:
            cfg = OmegaConf.load(EXPERIMENTS / "config/spatial_wovr_pi05_8gpu.yaml")
            OmegaConf.update(cfg, field, value)
            with self.assertRaises(ValueError):
                validate_batch(cfg)

    def test_kir_configuration_rejects_shadow_state_and_ungrouped_resets(self):
        for field, value in [
            ("env.train.use_libero_state_bridge", True),
            ("env.train.use_policy_state", True),
            ("env.train.auto_reset", True),
            ("env.train.ignore_terminations", True),
            ("env.train.kir_fraction", float("nan")),
        ]:
            cfg = OmegaConf.load(EXPERIMENTS / "config/spatial_wovr_pi05_8gpu.yaml")
            OmegaConf.update(cfg, field, value)
            with self.assertRaises(ValueError):
                validate_batch(cfg)

    def test_frame_success_mask_requires_per_action_training(self):
        for field, value in [
            ("algorithm.reward_type", "chunk_level"),
            ("algorithm.logprob_type", "chunk_level"),
        ]:
            cfg = OmegaConf.load(EXPERIMENTS / "config/spatial_wovr_pi05_8gpu.yaml")
            OmegaConf.update(cfg, field, value)
            with self.assertRaisesRegex(ValueError, "per-action"):
                validate_batch(cfg)

    def test_metadata_envelope_is_unwrapped_and_errors_fail_closed(self):
        self.assertEqual(
            metadata_payload({"ok": True, "data": {"chunk_size": 8}}), {"chunk_size": 8}
        )
        with self.assertRaises(ValueError):
            metadata_payload({"ok": False, "error": "loading"})
        with self.assertRaises(ValueError):
            metadata_payload({"ok": True, "data": []})

    def test_release_runtime_rejects_another_suite_or_camera_convention(self):
        from spatial_wovr_launch import WRIST_CONFIG

        runtime = {
            "cosmos_config": WRIST_CONFIG,
            "checkpoint_relative_path": "iter_000020000",
            "task_suite": "libero_spatial",
            "action_contract": "wovr13",
            "action_chunk": 8,
            "action_dim": 7,
            "primary_reference": True,
            "cross_view_primary_context": False,
            "runtime_camera_space": "rlinf_runtime_rot180",
        }
        validate_wrist_runtime(runtime)
        for key, value in [
            ("task_suite", "libero_10"),
            ("runtime_camera_space", "flipud"),
        ]:
            with self.assertRaises(ValueError):
                validate_wrist_runtime(dict(runtime, **{key: value}))

    def test_stale_or_wrong_checkpoint_service_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "iter_000020000"
            checkpoint.mkdir()
            (checkpoint.parent / "checkpoint_manifest.json").write_text(
                json.dumps({"tree_sha256": "release-digest"})
            )
            metadata = {
                "checkpoint_identity": {
                    "resolved_path": str(checkpoint),
                    "tree_sha256": "release-digest",
                },
                "wovr_action_contract": True,
                "chunk_size": 8,
                "action_dim": 7,
                "cross_view_primary_context": False,
                "checkpoint_load_audit": {"verified": True},
            }
            validate_server_metadata(metadata, checkpoint)
            for field, value in [
                ("chunk_size", 16),
                ("cross_view_primary_context", True),
                ("checkpoint_load_audit", {"verified": False}),
                (
                    "checkpoint_identity",
                    {"resolved_path": str(checkpoint), "tree_sha256": "old"},
                ),
            ]:
                changed = copy.deepcopy(metadata)
                changed[field] = value
                with self.assertRaises(ValueError):
                    validate_server_metadata(changed, checkpoint)

    def test_packaged_manifest_matches_actual_server_identity(self):
        helper = load_module(
            "serving_checkpoint_identity",
            PROJECT / "third_party/WorldPilot/cosmos_bridge/checkpoint_identity.py",
        )
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "iter_000002000"
            model = checkpoint / "model"
            model.mkdir(parents=True)
            (model / ".metadata").write_bytes(b"metadata")
            (model / "weights.distcp").write_bytes(b"updated weights")
            rows = [
                {"path": str(p.relative_to(checkpoint.parent)),
                 "size": p.stat().st_size,
                 "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                for p in sorted(model.iterdir())
            ]
            manifest_tree = hashlib.sha256(
                "".join(f"{row['sha256']}  {row['path']}\n" for row in rows).encode()
            ).hexdigest()
            (checkpoint.parent / "checkpoint_manifest.json").write_text(
                json.dumps({"tree_sha256": manifest_tree, "files": rows})
            )
            metadata = {
                "checkpoint_identity": helper.checkpoint_tree_identity(checkpoint),
                "wovr_action_contract": True, "chunk_size": 8, "action_dim": 7,
                "cross_view_primary_context": False,
                "checkpoint_load_audit": {"verified": True},
            }
            validate_server_metadata(metadata, checkpoint)
            (model / "weights.distcp").write_bytes(b"wrong weights")
            metadata["checkpoint_identity"] = helper.checkpoint_tree_identity(checkpoint)
            with self.assertRaisesRegex(ValueError, "checkpoint identity"):
                validate_server_metadata(metadata, checkpoint)

    def test_dcp_manifest_includes_missing_shards_instead_of_only_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "iter_000020000"
            (checkpoint / "model").mkdir(parents=True)
            (checkpoint / "model/.metadata").write_bytes(b"metadata")
            (root / "checkpoint_manifest.json").write_text(
                json.dumps(
                    {
                        "checkpoint": checkpoint.name,
                        "files": [
                            {"path": f"{checkpoint.name}/model/.metadata", "size": 8},
                            {
                                "path": f"{checkpoint.name}/model/__0_0.distcp",
                                "size": 4096,
                            },
                        ],
                    }
                )
            )
            records = wrist_records(checkpoint)
            self.assertEqual(len(records), 2)
            self.assertFalse(records[1][0].exists())

    def test_cleanup_targets_only_the_created_process_group(self):
        from unittest.mock import Mock

        process = Mock(pid=12345)
        process.poll.return_value = None
        with patch("spatial_wovr_launch.os.killpg") as kill:
            stop_process(process)
        self.assertEqual(kill.call_count, 1)
        self.assertEqual(kill.call_args.args[0], 12345)
        process.wait.assert_called_once_with(timeout=15)

    def test_actor_returns_cache_before_world_models_and_reserves_before_update(self):
        from spatial_wovr_diagnostics import MicroBatchDiagnostics

        events = []

        class BaseActor:
            async def sync_model_to_rollout(self):
                events.append("sync")

            def run_training(self):
                events.append("train")
                return {}

        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(
                synchronize=lambda device: events.append("synchronize"),
                empty_cache=lambda: events.append("release"),
            )
        )
        with patch.dict(
            sys.modules,
            {
                "torch": fake_torch,
                "spatial_grpo_actor": SimpleNamespace(
                    ReservedEmbodiedFSDPActor=BaseActor
                ),
            },
        ):
            module = load_module(
                "test_world_model_actor", EXPERIMENTS / "spatial_wovr_actor.py"
            )
        actor = module.WorldModelFSDPActor()
        actor.device = "cpu"
        actor.cfg = SimpleNamespace(runner={"memory_guard": {"enabled": False}})
        actor._cuda_cache_reservation = SimpleNamespace(
            retain=lambda phase: events.append(phase)
        )
        asyncio.run(actor.sync_model_to_rollout())
        with patch.object(
            module, "MicroBatchDiagnostics", return_value=MicroBatchDiagnostics()
        ):
            actor.run_training()
        self.assertEqual(
            events, ["synchronize", "release", "sync", "before_training", "train"]
        )
        events.clear()
        actor.cfg.runner["memory_guard"]["enabled"] = True
        asyncio.run(actor.sync_model_to_rollout())
        self.assertEqual(
            events, ["synchronize", "release", "sync", "synchronize", "release"]
        )


if __name__ == "__main__":
    unittest.main()
