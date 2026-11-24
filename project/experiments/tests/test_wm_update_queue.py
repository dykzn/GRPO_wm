"""No next GPU workload before successful completion; cleanup is run-specific."""

from __future__ import annotations

import json
import os
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from queue_spatial_wm_update import (
    completed_step,
    owned_remaining,
    predecessor_ready,
    run_queue,
    sha256,
    stages_for,
)
from stop_spatial_grpo import Process


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "resolved_config.yaml").write_text("runner: {max_steps: 20}\n")
        self.checkpoint = self.root / "experiment/checkpoints/global_step_20"
        self.weights = self.checkpoint / "actor/model_state_dict/full_weights.pt"
        self.weights.parent.mkdir(parents=True)
        self.weights.write_bytes(b"finished final policy")
        (self.root / "metrics.log").write_text("Global Step: 20/20\n")
        self.plan = {
            "grpo_run": str(self.root),
            "target_step": 20,
            "source_config_sha256": sha256(self.root / "resolved_config.yaml"),
            "policy_checkpoint": str(self.checkpoint),
            "tracked": [{"pid": 123, "start_ticks": 100, "role": "launcher"}],
        }

    def test_checkpoint_without_successful_exit_does_not_start_collection(self):
        self.assertFalse(predecessor_ready(self.plan))

    def test_failed_grpo_never_uses_even_existing_final_weights(self):
        (self.root / "exit_code").write_text("1\n")
        with self.assertRaisesRegex(RuntimeError, "exited with code 1"):
            predecessor_ready(self.plan)

    def test_early_successful_exit_does_not_fall_back_to_step_15(self):
        (self.root / "exit_code").write_text("0\n")
        (self.root / "metrics.log").write_text("Global Step: 15/20\n")
        with self.assertRaisesRegex(RuntimeError, "before the requested final step"):
            predecessor_ready(self.plan)

    def test_successful_exit_waits_for_launcher_cleanup(self):
        (self.root / "exit_code").write_text("0\n")
        with patch("queue_spatial_wm_update.process_alive", return_value=True):
            self.assertFalse(predecessor_ready(self.plan))
        with patch("queue_spatial_wm_update.process_alive", return_value=False):
            self.assertTrue(predecessor_ready(self.plan))

    def test_missing_final_weights_blocks_even_successful_exit(self):
        (self.root / "exit_code").write_text("0\n")
        self.weights.unlink()
        with self.assertRaisesRegex(RuntimeError, "no complete final policy"):
            predecessor_ready(self.plan)

    def test_changed_predecessor_config_is_rejected(self):
        (self.root / "resolved_config.yaml").write_text("runner: {max_steps: 50}\n")
        with self.assertRaisesRegex(RuntimeError, "configuration changed"):
            predecessor_ready(self.plan)

    def test_cleanup_protects_reused_pids_and_other_jobs(self):
        def process(pid, parent, start):
            return Process(pid, parent, os.geteuid(), start, "ray::Worker", (), {})

        processes = {
            100: process(100, 1, 9),  # Original owned worker.
            101: process(101, 100, 10),  # Its later child.
            200: process(200, 1, 80),  # Reused PID: original started at 8.
            201: process(201, 200, 81),
            300: process(300, 1, 90),  # Unrelated job.
        }
        plan = {
            "release_processes": [
                {"pid": 100, "start_ticks": 9},
                {"pid": 200, "start_ticks": 8},
            ]
        }
        with (
            patch("stop_spatial_grpo.snapshot", return_value=processes),
            patch("stop_spatial_grpo.protected_ancestors", return_value={999}),
        ):
            self.assertEqual(set(owned_remaining(plan)), {100, 101})

    def test_queued_pipeline_uses_final_policy_and_two_four_gpu_models(self):
        plan = dict(
            self.plan,
            output=str(self.root / "update"),
            python="/runtime/python",
            gpus=list(map(str, range(8))),
            seed=42,
            per_task=50,
            steps=2000,
            save_interval=200,
        )
        stages = stages_for(plan)
        self.assertEqual(
            [s["name"] for s in stages],
            [
                "pilot_collect",
                "pilot_prepare",
                "pilot_inspect",
                "backward_probe",
                "collect",
                "prepare",
                "inspect",
                "train",
                "package",
            ],
        )
        for stage in stages:
            cmd = stage["command"]
            if stage["name"] in {"pilot_collect", "collect"}:
                self.assertEqual(
                    cmd[cmd.index("--policy-checkpoint") + 1], str(self.checkpoint)
                )
            if stage["name"] in {"train", "backward_probe"}:
                self.assertEqual(cmd[cmd.index("--ranks-per-model") + 1], "4")
                self.assertEqual(cmd[cmd.index("--accumulation") + 1], "4")
        self.assertNotEqual(stages[3]["log"], stages[7]["log"])
        json.dumps(stages)

    def test_completed_steps_are_taken_from_logged_updates(self):
        (self.root / "metrics.log").write_text(
            "Global Step: 15/20\nGlobal Step: 16/20\n"
        )
        self.assertEqual(completed_step(self.root), 16)

    def test_failed_predecessor_never_spawns_a_pipeline_process(self):
        output = self.root / "queue"
        output.mkdir()
        plan = dict(self.plan, output=str(output))
        path = output / "plan.json"
        path.write_text(json.dumps(plan))
        (output / "state.json").write_text('{"status":"prepared"}')
        (self.root / "exit_code").write_text("1\n")
        handlers = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            with (
                patch("queue_spatial_wm_update.verify_plan"),
                patch("queue_spatial_wm_update.subprocess.Popen") as spawn,
            ):
                with self.assertRaisesRegex(RuntimeError, "exited with code 1"):
                    run_queue(path)
                spawn.assert_not_called()
            state = json.loads((output / "state.json").read_text())
            self.assertEqual(state["status"], "failed")
        finally:
            for s, handler in handlers.items():
                signal.signal(s, handler)

    def test_successful_handoff_runs_cpu_stages_in_order_without_cuda(self):
        output = self.root / "queue"
        output.mkdir()
        order = output / "order.txt"
        stages = []
        for name in ("collect", "train", "package"):
            code = (
                "import os; from pathlib import Path; "
                "assert os.environ['CUDA_VISIBLE_DEVICES']==''; "
                f"p=Path({str(order)!r}); "
                f"p.open('a').write({name!r}+'\\n')"
            )
            stages.append(
                {
                    "name": name,
                    "command": [sys.executable, "-c", code],
                    "log": str(output / f"{name}.log"),
                }
            )
        plan = dict(
            self.plan,
            output=str(output),
            stages=stages,
            steps=2,
            release_processes=[],
            gpus=["0"],
            min_free_gib=0,
        )
        path = output / "plan.json"
        path.write_text(json.dumps(plan))
        (output / "state.json").write_text('{"status":"prepared"}')
        (self.root / "exit_code").write_text("0\n")
        handlers = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            with (
                patch("queue_spatial_wm_update.verify_plan"),
                patch("queue_spatial_wm_update.process_alive", return_value=False),
                patch("queue_spatial_wm_update.release_predecessor"),
                patch("queue_spatial_wm_update.gpu_usage", return_value={"0": 0}),
            ):
                run_queue(path)
            self.assertEqual(
                order.read_text().splitlines(), ["collect", "train", "package"]
            )
            self.assertEqual(
                json.loads((output / "state.json").read_text())["status"], "complete"
            )
        finally:
            for s, handler in handlers.items():
                signal.signal(s, handler)


if __name__ == "__main__":
    unittest.main()
