"""CPU coverage for a single-camera evaluation and the four-to-eight handoff."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import queue_cosmos_first_wan8 as queue
import wm_eval
from wm_update_wan_reshard import validate_resume


class HandoffTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def checkpoint(self, step, *, exported=True, logged=True):
        path = self.root / f"step_{step:06d}"
        self.write(path / "complete.json", {"step": step, "world_size": 4, "identity": {"manifest": "split"}})
        (path / "state").mkdir()
        (path / "state/.metadata").touch()
        for rank in range(4):
            (path / f"rng_rank{rank}.pt").touch()
        if exported:
            self.write(path / "runtime/wm_update.json", {"step": step, "manifest_sha256": "split"})
            (path / "runtime/model-00001.safetensors").touch()
            (path / "runtime/Wan2.2_VAE.pth").touch()
        if logged:
            (self.root / "metrics.jsonl").write_text(json.dumps({"step": step}) + "\n")
        return path

    def test_only_complete_exported_logged_checkpoint_can_be_selected(self):
        expected = self.checkpoint(600)
        self.checkpoint(800, exported=False, logged=False)
        self.assertEqual(queue.latest_wan_checkpoint(self.root, "split"), expected)
        self.checkpoint(1000, logged=False)
        self.assertEqual(queue.latest_wan_checkpoint(self.root, "split"), expected)
        with self.assertRaises(ValueError):
            queue.latest_wan_checkpoint(self.root, "different-split")

    def test_resume_requires_same_settings_and_effective_batch(self):
        identity = {"manifest": "split", "seed": 42, "accumulation": 4, "lr": 1e-6}
        record = {"identity": identity, "world_size": 4}
        validate_resume(record, dict(identity, accumulation=2), 8)
        for candidate, world in [(dict(identity), 8), (dict(identity, accumulation=2, lr=1e-5), 8), (dict(identity, accumulation=2), 4)]:
            with self.assertRaises(ValueError):
                validate_resume(record, candidate, world)

    def test_restart_preserves_failure_history_and_validation_selection(self):
        old = self.root / "old"
        new = self.root / "retry"
        self.write(old / "state.json", {"status": "failed", "error": "read EIO"})
        self.write(old / "inputs/eval_manifest.json", {"manifest_sha256": "same-evaluation"})
        self.write(old / "inputs/reshard_cpu_proof.json", {"status": "passed"})
        self.write(old / "source_plan.json", {})
        previous = {"output": str(old), "source_plan": str(old / "source_plan.json"),
                    "source_supervisor": {"pid": 1, "start_ticks": 2},
                    "evaluation_manifest_sha256": "same-evaluation", "inputs": {}, "code": {},
                    "evaluation_summary": {"episodes": 100}, "effective_batch": 16}
        self.write(old / "plan.json", previous)
        args = argparse.Namespace(plan=old / "plan.json", output=new)
        with (patch.object(queue, "verify"), patch.object(queue, "verify_plan"),
              patch.object(queue, "process_alive", return_value=True),
              patch.object(queue, "cosmos_ready", return_value=True)):
            queue.restart(args)
        current = json.loads((new / "plan.json").read_text())
        self.assertEqual(current["restart_of"], str(old))
        self.assertEqual(current["evaluation_manifest_sha256"], "same-evaluation")
        self.assertEqual(current["effective_batch"], 16)
        self.assertIn("wm_eval_data.py", current["code"])
        self.assertEqual((new / "inputs/eval_manifest.json").read_bytes(), (old / "inputs/eval_manifest.json").read_bytes())
        self.assertEqual(json.loads((old / "state.json").read_text())["status"], "failed")
        self.assertFalse((new / "identity.json").exists())

    def test_restart_rejects_a_live_previous_attempt(self):
        self.write(self.root / "state.json", {"status": "failed"})
        self.write(self.root / "identity.json", {"pid": 1, "start_ticks": 2})
        self.write(self.root / "plan.json", {"output": str(self.root)})
        with (patch.object(queue, "process_alive", return_value=True),
              self.assertRaisesRegex(RuntimeError, "still alive")):
            queue.restart(argparse.Namespace(plan=self.root / "plan.json", output=self.root / "retry"))

    def test_cosmos_final_marker_alone_does_not_authorize_evaluation(self):
        plan = {"source_output": str(self.root), "steps": 2000,
                "training_manifest_sha256": "split", "controls": {"cosmos": {"pid": 1, "start_ticks": 2}},
                "source_supervisor": {"pid": 3, "start_ticks": 4}}
        self.write(queue.cosmos_source(plan) / "wm_update_complete.json", {"step": 2000, "identity": {"manifest": "split"}})
        self.write(self.root / "train/cosmos/metrics.jsonl", {"step": 2000})
        with patch.object(queue, "process_alive", return_value=True):
            self.assertFalse(queue.cosmos_ready(plan))
        with patch.object(queue, "process_alive", return_value=False):
            self.assertTrue(queue.cosmos_ready(plan))

    def test_evaluation_uses_only_selected_view_and_frozen_manifest(self):
        (self.root / "inputs").mkdir()
        (self.root / "inputs/eval_manifest.json").write_text('{"frozen":true}')
        plan = {"output": str(self.root), "python": "/python", "seed": 42}
        with patch.object(queue, "stage") as stage:
            queue.evaluate(plan, "cosmos", "updated", Path("/final/cosmos"), ["4", "5", "6", "7"], {})
        _, stage_name, command, env = stage.call_args.args
        self.assertEqual(stage_name, "cosmos_updated_metrics")
        self.assertEqual(command[command.index("--model") + 1], "cosmos")
        self.assertEqual(command[command.index("--gpus") + 1], "4,5,6,7")
        self.assertEqual(env["COSMOS_WRIST_CHECKPOINT"], "/final/cosmos")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "4,5,6,7")
        self.assertEqual((self.root / "video_metrics/cosmos/updated/eval_manifest.json").read_text(), '{"frozen":true}')

    def test_single_model_run_does_not_start_the_other_model(self):
        args = argparse.Namespace(model="cosmos", gpus="4,5,6,7", workers_per_gpu=1,
                                  output=self.root, manifest=self.root / "manifest.json", steps=5,
                                  seed=42, warmup=3, metric_batch=8)
        manifest = {"manifest_sha256": "split", "episodes": [{"trajectory_id": "a"}]}
        with (patch.object(wm_eval, "readiness", return_value={"ready": True}) as ready,
              patch.object(wm_eval, "prepare_code_runtime"),
              patch.object(wm_eval, "supervise") as supervise,
              patch.object(wm_eval, "prediction_commands", return_value=[]) as predict,
              patch.object(wm_eval, "completed_predictions", return_value={}),
              patch.object(wm_eval, "cosmos_probe_command", return_value=None),
              patch.object(wm_eval.subprocess, "run"),
              patch.dict(wm_eval.os.environ, {"COSMOS_PYTHON": "/cosmos", "RLINF_ENV": "/rlinf"})):
            wm_eval.run(args, manifest)
        ready.assert_called_once_with(manifest, ["cosmos"])
        self.assertEqual([call.args[1] for call in predict.call_args_list], ["cosmos"])
        command = supervise.call_args.args[0][0][0]
        self.assertEqual(command[command.index("--model") + 1], "cosmos")


if __name__ == "__main__":
    unittest.main()
