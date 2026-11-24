"""The video comparison must wait for training and use identical held-out data."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import queue_wm_video_metrics as queue


class VideoQueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.training = self.root / "training"
        self.output = self.root / "evaluation"
        self.training.mkdir()
        self.plan = {
            "output": str(self.output),
            "training_output": str(self.training),
            "training_identity": {"pid": 1, "start_ticks": 2},
            "target_step": 2000,
            "pair": str(self.training / "pair_2000"),
            "training_manifest_sha256": "full-manifest",
            "evaluation_manifest_sha256": "held-out-manifest",
            "python": sys.executable,
            "gpus": list(map(str, range(8))),
            "seed": 20261021,
            "protocol_note": "unit test fixture",
        }

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def complete_source(self):
        self.write(self.training / "state.json", {"status": "complete"})
        (self.training / "exit_code").write_text("0\n")
        self.write(
            self.training / "pair_2000/pair.json",
            {
                "wan_step": 2000,
                "cosmos_step": 2000,
                "manifest_sha256": "full-manifest",
            },
        )

    def test_active_training_and_final_cleanup_do_not_authorize_evaluation(self):
        self.write(
            self.training / "state.json", {"status": "running", "stage": "train"}
        )
        with patch.object(queue, "process_alive", return_value=True):
            self.assertFalse(queue.upstream_ready(self.plan))
            self.complete_source()
            self.assertFalse(queue.upstream_ready(self.plan))

    def test_success_requires_zero_exit_complete_target_and_original_split(self):
        self.complete_source()
        with patch.object(queue, "process_alive", return_value=False):
            self.assertTrue(queue.upstream_ready(self.plan))
            self.write(
                self.training / "pair_2000/pair.json",
                {
                    "wan_step": 2000,
                    "cosmos_step": 1800,
                    "manifest_sha256": "full-manifest",
                },
            )
            with self.assertRaisesRegex(RuntimeError, "unexpected update steps"):
                queue.upstream_ready(self.plan)
            self.write(
                self.training / "pair_2000/pair.json",
                {
                    "wan_step": 2000,
                    "cosmos_step": 2000,
                    "manifest_sha256": "another-split",
                },
            )
            with self.assertRaisesRegex(
                RuntimeError, "different training/validation split"
            ):
                queue.upstream_ready(self.plan)

    def test_upstream_failure_or_abrupt_exit_stops_evaluation(self):
        self.write(
            self.training / "state.json", {"status": "failed", "error": "worker error"}
        )
        with self.assertRaisesRegex(RuntimeError, "did not succeed"):
            queue.upstream_ready(self.plan)
        self.write(self.training / "state.json", {"status": "running"})
        with (
            patch.object(queue, "process_alive", return_value=False),
            self.assertRaisesRegex(RuntimeError, "before successful completion"),
        ):
            queue.upstream_ready(self.plan)

    def test_summary_rejects_different_manifest_and_nonfinite_metrics(self):
        for version in ("baseline", "updated"):
            rows = [
                {
                    "model": model,
                    "view": "primary" if model == "wan" else "wrist",
                    "episodes": 100,
                    "frames": 14000,
                    "manifest_sha256": "held-out-manifest",
                    **{key: 1.0 for key in queue.METRICS},
                }
                for model in ("wan", "cosmos")
            ]
            self.write(self.output / version / "metrics.json", rows)
        self.write(self.output / "comparison/comparison.json", {"fixture": True})
        self.assertEqual(len(queue.write_summary(self.plan)), 4)
        candidate = json.loads((self.output / "updated/metrics.json").read_text())
        candidate[0]["manifest_sha256"] = "different-video-set"
        self.write(self.output / "updated/metrics.json", candidate)
        with self.assertRaisesRegex(ValueError, "frozen held-out"):
            queue.write_summary(self.plan)
        candidate[0]["manifest_sha256"] = "held-out-manifest"
        candidate[0]["FID"] = float("nan")
        self.write(self.output / "updated/metrics.json", candidate)
        with self.assertRaisesRegex(ValueError, "finite"):
            queue.write_summary(self.plan)


if __name__ == "__main__":
    unittest.main()
