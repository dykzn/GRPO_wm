"""Formal WM emails must not claim success for missing final outputs."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import watch_wm_update_email as watcher


class WMOutcomeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.plan = {"output": str(self.root), "per_task": 50, "steps": 2000}

    def write(self, path, value):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value))

    def test_live_collection_is_pending(self):
        self.write("state.json", {"status": "running", "stage": "collect"})
        self.write("worker_identity.json", {"pid": 1, "start_ticks": 2})
        with patch.object(watcher, "process_alive", return_value=True):
            self.assertIsNone(watcher.terminal_result(self.plan))

    def test_success_requires_all_500_episodes_and_both_final_models(self):
        self.write("state.json", {"status": "complete"})
        self.assertEqual(watcher.terminal_result(self.plan)["outcome"], "failed")
        self.write("collection/collection.json", {"summary": {"episodes": 500}})
        self.write("pair_2000/pair.json", {"wan_step": 2000, "cosmos_step": 1800})
        self.assertEqual(watcher.terminal_result(self.plan)["outcome"], "failed")
        self.write("pair_2000/pair.json", {"wan_step": 2000, "cosmos_step": 2000})
        self.assertEqual(watcher.terminal_result(self.plan)["outcome"], "complete")

    def test_failure_before_training_is_notified(self):
        self.write(
            "state.json", {"status": "failed", "error": "Collection worker exited 1"}
        )
        self.assertEqual(
            watcher.terminal_result(self.plan)["error"], "Collection worker exited 1"
        )

    def test_abrupt_exit_is_detected_and_final_state_is_reread(self):
        self.write("state.json", {"status": "running", "stage": "train"})
        self.write("worker_identity.json", {"pid": 1, "start_ticks": 2})
        with patch.object(watcher, "process_alive", return_value=False):
            self.assertEqual(watcher.terminal_result(self.plan)["outcome"], "failed")

        def just_failed(identity):
            self.write(
                "state.json", {"status": "failed", "error": "Specific worker error"}
            )
            return False

        with patch.object(watcher, "process_alive", side_effect=just_failed):
            self.assertEqual(
                watcher.terminal_result(self.plan)["error"], "Specific worker error"
            )


if __name__ == "__main__":
    unittest.main()
