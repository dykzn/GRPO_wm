"""Regression checks for observed IO/config failures and exact continuation."""

from __future__ import annotations

import contextlib
import io
import json
import os
import signal
import sys
import tempfile
import types
import unittest
from functools import partial
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from queue_spatial_wm_update import (
    run_queue,
    sha256,
    verify_saved_policy,
    wait_for_gpus,
)
from wm_update_cosmos import (
    config_snapshot,
    restore_master_weights,
    save_master_weights,
)
from wm_update_data import inventory, write_episode


class HardeningTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_partial_callable_roundtrips_to_safe_config_yaml(self):
        cfg = {
            "linear": partial(torch.nn.Linear, 3, bias=False),
            "dtype": torch.bfloat16,
        }
        value = yaml.safe_load(yaml.safe_dump(config_snapshot(cfg)))
        self.assertEqual(value["linear"]["_target_"], "functools.partial")
        self.assertEqual(value["linear"]["func"], "torch.nn.modules.linear.Linear")
        self.assertEqual(value["linear"]["args"], [3])
        self.assertEqual(value["linear"]["keywords"], {"bias": False})
        self.assertEqual(value["dtype"], "torch.bfloat16")

    def optimizer(self):
        master = torch.tensor([1.001234, -0.500123], dtype=torch.float32)
        p = torch.nn.Parameter(master.to(torch.bfloat16))
        optimizer = types.SimpleNamespace(
            master_weights=True,
            param_groups=[{"params": [p]}],
            param_groups_master=[{"params": [master]}],
        )
        return optimizer, master

    def test_fp32_master_is_restored_exactly_instead_of_bf16_rounded_copy(self):
        opt, expected = self.optimizer()
        path = self.root / "master_weights.pt"
        save_master_weights(path, opt)
        self.assertFalse(
            torch.equal(expected, opt.param_groups[0]["params"][0].float())
        )
        opt.param_groups_master = None
        restore_master_weights(path, opt)
        torch.testing.assert_close(
            opt.param_groups_master[0]["params"][0], expected, rtol=0, atol=0
        )
        self.assertEqual(opt.param_groups_master[0]["params"][0].dtype, torch.float32)
        self.assertFalse(torch.cuda.is_initialized())

    def test_master_shape_mismatch_is_rejected(self):
        opt, _ = self.optimizer()
        path = self.root / "master_weights.pt"
        save_master_weights(path, opt)
        opt.param_groups[0]["params"][0] = torch.nn.Parameter(
            torch.zeros(3, dtype=torch.bfloat16)
        )
        with self.assertRaisesRegex(ValueError, "shape or precision"):
            restore_master_weights(path, opt)

    def test_missing_initialized_master_is_not_saved_as_complete(self):
        opt, _ = self.optimizer()
        opt.param_groups_master = None
        with self.assertRaisesRegex(RuntimeError, "not initialized"):
            save_master_weights(self.root / "master_weights.pt", opt)

    def test_waits_until_all_cards_are_free_without_starting_processes(self):
        plan = {
            "gpus": ["0", "1"],
            "wait_for_gpus": True,
            "poll_seconds": 1,
            "target_step": 15,
            "startup_mode": "saved_policy",
        }
        with (
            patch(
                "queue_spatial_wm_update.gpu_usage",
                side_effect=[
                    {"0": 10000, "1": 0},
                    {"0": 0, "1": 2000},
                    {"0": 0, "1": 0},
                ],
            ) as usage,
            patch("queue_spatial_wm_update.time.sleep") as sleep,
            patch("queue_spatial_wm_update.gpu_compute_pids", return_value={}),
            patch("queue_spatial_wm_update.subprocess.Popen") as spawn,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            wait_for_gpus(plan, self.root)
        self.assertEqual(usage.call_count, 3)
        self.assertEqual(sleep.call_count, 2)
        spawn.assert_not_called()
        state = json.loads((self.root / "state.json").read_text())
        self.assertEqual(state["status"], "waiting_for_gpus")

    def test_immediate_start_without_wait_flag_rejects_busy_gpu(self):
        plan = {"gpus": ["0"], "startup_mode": "saved_policy"}
        with (
            patch("queue_spatial_wm_update.gpu_usage", return_value={"0": 50000}),
            self.assertRaisesRegex(RuntimeError, "occupied"),
        ):
            wait_for_gpus(plan, self.root)

    def test_saved_policy_is_bound_to_selected_node_and_immutable_weights(self):
        cfg = self.root / "config.yaml"
        cfg.write_text("env: {}")
        checkpoint = self.root / "global_step_15"
        weights = checkpoint / "actor/model_state_dict/full_weights.pt"
        weights.parent.mkdir(parents=True)
        weights.write_bytes(b"selected saved weights")
        stat = weights.stat()
        plan = {
            "expected_node_ip": "172.44.19.145",
            "execution_boot_id": Path("/proc/sys/kernel/random/boot_id")
            .read_text()
            .strip(),
            "source_config_path": str(cfg),
            "source_config_sha256": sha256(cfg),
            "policy_checkpoint": str(checkpoint),
            "policy_weight_identity": {
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": sha256(weights),
            },
        }
        with patch("queue_spatial_wm_update.subprocess.run") as run:
            run.return_value.stdout = "172.44.19.145"
            verify_saved_policy(plan)
            run.return_value.stdout = "172.44.19.146"
            with self.assertRaisesRegex(RuntimeError, "different execution node"):
                verify_saved_policy(plan)
            run.return_value.stdout = "172.44.19.145"
            weights.write_bytes(b"different saved policy")
            os.utime(weights, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            self.assertEqual(weights.stat().st_size, stat.st_size)
            with self.assertRaisesRegex(RuntimeError, "weights changed"):
                verify_saved_policy(plan)

    def test_retry_reuses_successful_stages_without_waiting_for_remote_grpo(self):
        marker = self.root / "ran.txt"
        stages = [
            {
                "name": "pilot_collect",
                "command": [sys.executable, "-c", "raise SystemExit(99)"],
                "log": str(self.root / "old.log"),
            },
            {
                "name": "prepare",
                "command": [
                    sys.executable,
                    "-c",
                    f"from pathlib import Path; Path({str(marker)!r}).write_text('ran')",
                ],
                "log": str(self.root / "new.log"),
            },
        ]
        plan = {
            "output": str(self.root),
            "startup_mode": "saved_policy",
            "expected_node_ip": "172.44.19.145",
            "target_step": 15,
            "gpus": ["0"],
            "min_free_gib": 0,
            "steps": 2,
            "stages": stages,
        }
        path = self.root / "plan.json"
        path.write_text(json.dumps(plan))
        (self.root / "state.json").write_text('{"status":"prepared"}')
        (self.root / "events.jsonl").write_text(
            json.dumps({"status": "stage_complete", "stage": "pilot_collect"}) + "\n"
        )
        handlers = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            with (
                patch("queue_spatial_wm_update.verify_plan"),
                patch("queue_spatial_wm_update.verify_saved_policy"),
                patch(
                    "queue_spatial_wm_update.predecessor_ready",
                    side_effect=AssertionError("must not wait for remote job"),
                ),
                patch("queue_spatial_wm_update.release_predecessor") as cleanup,
                patch("queue_spatial_wm_update.gpu_usage", return_value={"0": 0}),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                run_queue(path)
                cleanup.assert_not_called()
            self.assertEqual(marker.read_text(), "ran")
            self.assertEqual(
                json.loads((self.root / "state.json").read_text())["status"], "complete"
            )
        finally:
            for signum, handler in handlers.items():
                signal.signal(signum, handler)

    def test_hdf5_io_survives_native_path_driver_errors(self):
        path = self.root / "episode.hdf5"
        real_file = h5py.File

        def buffered_only(value, *args, **kwargs):
            if isinstance(value, (str, Path)):
                raise OSError(5, "simulated native file-driver IO failure")
            return real_file(value, *args, **kwargs)

        actions = np.zeros((8, 7), dtype=np.float32)
        actions[:, -1] = -1
        frames = [np.zeros((8, 8, 3), dtype=np.uint8)] * 9
        metadata = {
            "episode_id": "regression",
            "task_id": 0,
            "task": "task",
            "suite": "libero_spatial",
            "policy_id": "policy",
            "initial_state_id": 0,
            "seed": 0,
            "success": False,
        }
        with patch("wm_update_data.h5py.File", side_effect=buffered_only):
            write_episode(
                path,
                primary=frames,
                wrist=frames,
                actions=actions,
                states=np.zeros((9, 5)),
                proprio=np.zeros((9, 9)),
                successes=np.zeros(8, dtype=bool),
                rewards=np.zeros(8),
                dones=np.zeros(8, dtype=bool),
                metadata=metadata,
            )
            row = inventory(path, verify_rgb=True)
        self.assertEqual(row["actions"], 8)
        self.assertFalse(row["success"])

    def test_low_memory_foreign_compute_process_is_still_busy(self):
        plan = {
            "gpus": ["0"],
            "wait_for_gpus": True,
            "poll_seconds": 1,
            "target_step": 15,
            "startup_mode": "saved_policy",
        }
        with (
            patch("queue_spatial_wm_update.gpu_usage", return_value={"0": 4}),
            patch(
                "queue_spatial_wm_update.gpu_compute_pids",
                side_effect=[{"0": [99]}, {}],
            ) as active,
            patch("queue_spatial_wm_update.time.sleep") as sleep,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            wait_for_gpus(plan, self.root)
        self.assertEqual(active.call_count, 2)
        sleep.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()
