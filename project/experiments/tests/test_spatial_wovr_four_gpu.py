"""Exercise the real four-GPU shell entry points without launching CUDA jobs."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

EXPERIMENTS = Path(__file__).resolve().parents[1]
PROJECT = EXPERIMENTS.parent
sys.path.insert(0, str(EXPERIMENTS))

from spatial_wovr_launch import make_endpoints, placement_world_size, validate_batch


class FourGpuTests(unittest.TestCase):
    def capture_wrapper(self, script, *, temp_root=None, overrides=(), env_overrides=None):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary = root / "bin"
            binary.mkdir()
            shim = binary / "python"
            shim.write_text(
                "#!/usr/bin/python3\n"
                "import json, os, sys\n"
                "from pathlib import Path\n"
                'keys = ("SPATIAL_WOVR_GPUS", "SPATIAL_WOVR_PLACEMENT", "SPATIAL_WOVR_OUTPUT", "SPATIAL_WOVR_RUNTIME", "SPATIAL_WOVR_FAST_TRANSFER", "SPATIAL_WATCH_RUN_ROOT", "TMPDIR", "RAY_TMPDIR")\n'
                'Path(os.environ["CAPTURE_PATH"]).write_text(json.dumps({"argv": sys.argv[1:], "env": {key: os.environ.get(key) for key in keys}}))\n'
            )
            shim.chmod(0o700)
            (binary / "python3").symlink_to(shim)
            capture = root / "capture.json"
            environment = dict(os.environ)
            for key in (
                "SPATIAL_WOVR_GPUS",
                "SPATIAL_WOVR_OUTPUT",
                "SPATIAL_WATCH_RUN",
                "SPATIAL_WATCH_RUN_ROOT",
                "SPATIAL_WOVR_TMP_ROOT",
                "SPATIAL_WOVR_FAST_TRANSFER",
                "SPATIAL_WOVR_FAST_COSMOS_BATCH",
                "SPATIAL_WOVR_FAST_VIDEO_STRIDE",
            ):
                environment.pop(key, None)
            environment.update(
                RLINF_ENV=str(root),
                CAPTURE_PATH=str(capture),
                SPATIAL_WOVR_RUNTIME=str(root / "runtime"),
                SPATIAL_WATCH_LOG_ROOT=str(root / "watch_logs"),
                PATH=str(binary) + os.pathsep + os.environ["PATH"],
            )
            if temp_root is not None:
                environment["SPATIAL_WOVR_TMP_ROOT"] = str(temp_root)
            if env_overrides:
                environment.update(env_overrides)
            subprocess.run(
                ["bash", str(EXPERIMENTS / script), *overrides],
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=True,
                timeout=15,
            )
            value = json.loads(capture.read_text())
            if "backward" in script or "20step" in script:
                # The shell bootstrap creates this empty directory before Python.
                Path(value["env"]["TMPDIR"]).rmdir()
            return value

    def composed_config(self, capture):
        environment = {
            "PROJECT_ROOT": str(PROJECT),
            "RLINF_BASE_ROOT": "/data41/qianb2608/wan/RLinf_opd",
            "SPATIAL_WOVR_PLACEMENT": capture["env"]["SPATIAL_WOVR_PLACEMENT"],
        }
        with (
            patch.dict(os.environ, environment),
            initialize_config_dir(
                config_dir=str(EXPERIMENTS / "config"), version_base="1.1"
            ),
        ):
            cfg = compose(
                config_name="spatial_wovr_pi05_8gpu", overrides=capture["argv"][1:]
            )
            cfg.cluster.component_placement = OmegaConf.to_container(
                cfg.cluster.component_placement, resolve=True
            )
            validate_batch(cfg)
        return cfg

    def test_twenty_step_entry_preserves_samples_per_update_and_per_gpu_load(self):
        capture = self.capture_wrapper("run_spatial_wovr_20step_4gpu.sh")
        self.assertEqual(capture["env"]["SPATIAL_WOVR_GPUS"], "0,1,2,3")
        self.assertIn(
            "spatial_wovr20_pi05_12000_4gpu/", capture["env"]["SPATIAL_WOVR_OUTPUT"]
        )
        cfg = self.composed_config(capture)
        self.assertEqual(placement_world_size(cfg), 4)
        self.assertEqual(validate_batch(cfg), 30720)
        self.assertEqual(cfg.env.train.total_num_envs // 4, 16)
        self.assertEqual(cfg.env.train.rollout_epoch, 16)
        self.assertEqual(cfg.actor.micro_batch_size, 32)
        self.assertEqual(cfg.actor.global_batch_size, 2048)
        self.assertEqual(
            cfg.actor.global_batch_size // 4 // cfg.actor.micro_batch_size, 16
        )
        self.assertEqual(cfg.algorithm.group_size, 8)
        self.assertEqual(cfg.runner.max_steps, 20)
        self.assertEqual(cfg.runner.save_interval, 5)
        self.assertEqual(cfg.runner.val_check_interval, 5)
        self.assertEqual(
            list(cfg.actor.fsdp_config.save_training_state_at_steps), [10, 20]
        )
        self.assertEqual(cfg.env.eval.total_num_envs // 4, 8)
        self.assertEqual(cfg.env.eval.rollout_epoch, 16)
        cfg.env.eval.rollout_epoch = 15
        with self.assertRaisesRegex(ValueError, "16 waves"):
            validate_batch(cfg)

    def test_peak_probe_collects_one_global_batch_and_sixteen_backwards_per_rank(self):
        capture = self.capture_wrapper("run_spatial_backward_peak_4gpu.sh")
        cfg = self.composed_config(capture)
        self.assertEqual(validate_batch(cfg), 2048)
        self.assertEqual(cfg.env.train.total_num_envs // 4, 16)
        self.assertEqual(cfg.env.train.rollout_epoch, 2)
        self.assertEqual(
            cfg.actor.global_batch_size // 4 // cfg.actor.micro_batch_size, 16
        )
        self.assertEqual(cfg.runner.max_steps, 1)
        self.assertLess(cfg.runner.val_check_interval, 0)
        self.assertLess(cfg.runner.save_interval, 0)
        self.assertEqual(cfg.actor.gpu_memory_profile.expected_optimizer_updates, 1)

    def test_fast_twenty_step_entry_preserves_protocol_and_opts_into_serial_changes(self):
        capture = self.capture_wrapper("run_spatial_wovr_20step_fast_4gpu.sh")
        cfg = self.composed_config(capture)
        self.assertEqual(Path(capture["env"]["TMPDIR"]).parent, PROJECT.parent / "r")
        self.assertEqual(capture["env"]["RAY_TMPDIR"], capture["env"]["TMPDIR"])
        self.assertIn("spatial_wovr20_fast_pi05_12000_4gpu/", capture["env"]["SPATIAL_WOVR_OUTPUT"])
        self.assertEqual(capture["env"]["SPATIAL_WOVR_FAST_TRANSFER"], "1")
        self.assertEqual(capture["env"]["SPATIAL_WOVR_GPUS"], "0,1,2,3")
        self.assertEqual(validate_batch(cfg), 30720)
        self.assertEqual(cfg.env.train.cosmos_batch_size, 4)
        self.assertFalse(cfg.env.train.cosmos_wrist_parallel_generation)
        self.assertTrue(cfg.env.train.fast_frame_transfer)
        self.assertTrue(cfg.env.train.skip_successful_env_generation)
        self.assertEqual(cfg.env.train.video_cfg.record_every_n_episodes, 4)
        self.assertTrue(cfg.env.eval.video_cfg.save_video)
        self.assertEqual(cfg.env.eval.video_cfg.get("record_every_n_episodes", 1), 1)
        self.assertEqual(cfg.env.train.num_inference_steps, 5)
        self.assertEqual(cfg.env.train.cosmos_future_denoising_steps, 5)
        self.assertEqual(cfg.algorithm.group_size, 8)
        self.assertEqual(cfg.actor.global_batch_size, 2048)
        self.assertEqual(cfg.env.train.total_num_envs, 64)
        self.assertEqual(cfg.env.train.rollout_epoch, 16)
        self.assertEqual(cfg.env.train.max_steps_per_rollout_epoch, 240)
        self.assertEqual(cfg.runner.max_steps, 20)
        self.assertEqual(cfg.runner.save_interval, 5)
        self.assertEqual(cfg.runner.val_check_interval, 5)
        self.assertEqual(list(cfg.actor.fsdp_config.save_training_state_at_steps), [10, 20])

    def test_fast_peak_probe_has_one_update_with_matching_generation_settings(self):
        capture = self.capture_wrapper("run_spatial_backward_peak_fast_4gpu.sh")
        cfg = self.composed_config(capture)
        self.assertEqual(Path(capture["env"]["TMPDIR"]).parent, PROJECT.parent / "r")
        self.assertEqual(capture["env"]["RAY_TMPDIR"], capture["env"]["TMPDIR"])
        self.assertIn("spatial_backward_peak_fast_4gpu/", capture["env"]["SPATIAL_WOVR_OUTPUT"])
        self.assertEqual(validate_batch(cfg), 2048)
        self.assertEqual(cfg.env.train.cosmos_batch_size, 4)
        self.assertTrue(cfg.env.train.skip_successful_env_generation)
        self.assertTrue(cfg.env.train.fast_frame_transfer)
        self.assertFalse(cfg.env.train.cosmos_wrist_parallel_generation)
        self.assertEqual(cfg.env.train.video_cfg.record_every_n_episodes, 4)
        self.assertEqual(cfg.actor.gpu_memory_profile.expected_optimizer_updates, 1)
        self.assertEqual(cfg.actor.global_batch_size // 4 // cfg.actor.micro_batch_size, 16)
        self.assertLess(cfg.runner.val_check_interval, 0)
        self.assertLess(cfg.runner.save_interval, 0)

    def test_fast_entry_allows_explicit_paths_batch_and_video_stride_overrides(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, runtime = Path(temporary) / "output", Path(temporary) / "runtime"
            capture = self.capture_wrapper(
                "run_spatial_wovr_20step_fast_4gpu.sh",
                env_overrides={
                    "SPATIAL_WOVR_OUTPUT": str(output),
                    "SPATIAL_WOVR_RUNTIME": str(runtime),
                    "SPATIAL_WOVR_FAST_COSMOS_BATCH": "2",
                    "SPATIAL_WOVR_FAST_VIDEO_STRIDE": "8",
                },
                overrides=("env.train.video_cfg.record_every_n_episodes=2",),
            )
            cfg = self.composed_config(capture)
            self.assertEqual(capture["env"]["SPATIAL_WOVR_OUTPUT"], str(output))
            self.assertEqual(capture["env"]["SPATIAL_WOVR_RUNTIME"], str(runtime))
            self.assertEqual(cfg.env.train.cosmos_batch_size, 2)
            self.assertEqual(cfg.env.train.video_cfg.record_every_n_episodes, 2)

    def test_fast_entries_honor_explicit_short_temporary_root(self):
        with tempfile.TemporaryDirectory(prefix="wovr-", dir="/tmp") as temporary:
            custom_root = Path(temporary) / "r"
            for script in (
                "run_spatial_wovr_20step_fast_4gpu.sh",
                "run_spatial_backward_peak_fast_4gpu.sh",
            ):
                with self.subTest(script=script):
                    capture = self.capture_wrapper(script, temp_root=custom_root)
                    self.assertEqual(Path(capture["env"]["TMPDIR"]).parent, custom_root.resolve())
                    self.assertEqual(capture["env"]["RAY_TMPDIR"], capture["env"]["TMPDIR"])
                    self.assertEqual(list(custom_root.iterdir()), [])

    def test_ordinary_entry_keeps_generation_transfer_and_video_defaults(self):
        cfg = self.composed_config(self.capture_wrapper("run_spatial_wovr_20step_4gpu.sh"))
        self.assertEqual(cfg.env.train.cosmos_batch_size, 2)
        self.assertFalse(cfg.env.train.skip_successful_env_generation)
        self.assertFalse(cfg.env.train.fast_frame_transfer)
        self.assertFalse(cfg.env.train.cosmos_wrist_parallel_generation)
        self.assertEqual(cfg.env.train.video_cfg.record_every_n_episodes, 1)

    def test_active_generation_rejects_unsupported_state_and_termination_modes(self):
        baseline = self.composed_config(self.capture_wrapper("run_spatial_wovr_20step_fast_4gpu.sh"))
        for field, value in (
            ("mask_after_first_success", False),
            ("auto_reset", True),
            ("ignore_terminations", True),
            ("use_libero_state_bridge", True),
            ("use_policy_state", True),
            ("cosmos_value_enabled", True),
            ("cosmos_use_predicted_proprio", True),
            ("cosmos_wrist_cross_view_primary_history", True),
        ):
            with self.subTest(field=field):
                cfg = OmegaConf.create(OmegaConf.to_container(baseline, resolve=False))
                cfg.env.train[field] = value
                with self.assertRaisesRegex(ValueError, "Skipping successful environments"):
                    validate_batch(cfg)

    def test_fast_batch_and_video_stride_must_be_positive_integers(self):
        baseline = self.composed_config(self.capture_wrapper("run_spatial_wovr_20step_fast_4gpu.sh"))
        for field in ("cosmos_batch_size", "video_cfg.record_every_n_episodes"):
            for value in (0, -1, True, 1.5):
                with self.subTest(field=field, value=value):
                    cfg = OmegaConf.create(OmegaConf.to_container(baseline, resolve=False))
                    OmegaConf.update(cfg.env.train, field, value)
                    with self.assertRaisesRegex(ValueError, "positive integer"):
                        validate_batch(cfg)

    def test_four_gpu_watchdog_selects_the_four_gpu_run_root(self):
        capture = self.capture_wrapper("run_spatial_wovr_watchdog_4gpu.sh")
        self.assertIn("--latest-run-root", capture["argv"])
        self.assertEqual(
            capture["argv"][-1],
            "/data12/qianb2608/grpo_runs/spatial_wovr20_pi05_12000_4gpu",
        )

    def test_twenty_step_entry_uses_and_cleans_custom_short_temporary_root(self):
        with tempfile.TemporaryDirectory(prefix="wovr-", dir="/tmp") as temporary:
            root = Path(temporary) / "r"
            capture = self.capture_wrapper(
                "run_spatial_wovr_20step_4gpu.sh", temp_root=root
            )
            directory = Path(capture["env"]["TMPDIR"])
            self.assertEqual(directory.parent, root.resolve())
            self.assertTrue(directory.name.startswith("w-"))
            self.assertEqual(capture["env"]["RAY_TMPDIR"], str(directory))
            socket = directory / (
                "ray/session_0000-00-00_00-00-00_000000_2147483647"
                "/sockets/plasma_store"
            )
            self.assertLess(len(os.fsencode(socket)), 108)
            self.assertFalse(directory.exists())
            self.assertEqual(list(root.iterdir()), [])

    def test_twenty_step_entry_rejects_root_that_exceeds_ray_socket_limit(self):
        with tempfile.TemporaryDirectory(prefix="wovr-", dir="/tmp") as temporary:
            root = Path(temporary) / ("x" * 30)
            with self.assertRaises(subprocess.CalledProcessError) as error:
                self.capture_wrapper("run_spatial_wovr_20step_4gpu.sh", temp_root=root)
            self.assertIn("SPATIAL_WOVR_TMP_ROOT is too long", error.exception.output)
            self.assertEqual(list(root.iterdir()), [])

    def test_endpoint_count_and_port_budget_follow_four_visible_devices(self):
        endpoints = make_endpoints("3,1,2,0", 65532)
        self.assertEqual(
            [item["port"] for item in endpoints], [65532, 65533, 65534, 65535]
        )
        for devices in ("0,0,1,2", "0,1,2", "0,1,2,a"):
            with self.assertRaises(ValueError):
                make_endpoints(devices, 18059)
        with self.assertRaises(ValueError):
            make_endpoints("0,1,2,3", 65533)


if __name__ == "__main__":
    unittest.main()
