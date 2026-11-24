"""Verify snapshot cleanup does not target evaluation or recycled PIDs."""

import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stop_snapshot_gpu_workers import rejection_reason, snapshot_pids
from stop_spatial_grpo import Process


SUFFIX = "PilotSftWorker.run_training"


def worker(pid=100, uid=None, started=1000, argv=None):
    return Process(pid, 1, os.geteuid() if uid is None else uid, started,
                   "ray::WorldPilot", tuple(argv or ["ray::WorldPilotSftWorker.run_training"]), {})


class SnapshotCleanupTests(unittest.TestCase):
    def test_only_matching_compute_rows_are_selected(self):
        text = """|    0   N/A  N/A      100      C   ...ldPilotSftWorker.run_training      42190MiB |
|    1   N/A  N/A      101      C   ...ldPilotSftWorker.run_training      42174MiB |
|    0   N/A  N/A      102      C   python wm_eval.py predict              9000MiB |
|    2   N/A  N/A      103      C   another_training                     42000MiB |"""
        self.assertEqual(snapshot_pids(text, SUFFIX), {100, 101})

    def test_matching_old_worker_is_eligible(self):
        self.assertIsNone(rejection_reason(worker(), {100}, SUFFIX, cutoff=120, boot=100, ticks=100))

    def test_missing_other_user_inactive_and_recycled_pids_are_rejected(self):
        for process, active in [(None, {100}), (worker(uid=os.geteuid()+1), {100}),
                                (worker(), set()), (worker(started=3000), {100})]:
            self.assertIsNotNone(rejection_reason(process, active, SUFFIX, cutoff=120, boot=100, ticks=100))

    def test_live_name_must_still_match_and_eval_is_protected(self):
        for argv in [["python", "wm_eval.py", "--worker", SUFFIX], ["ray::OtherWorker"]]:
            self.assertIsNotNone(rejection_reason(worker(argv=argv), {100}, SUFFIX, cutoff=120, boot=100, ticks=100))


if __name__ == "__main__":
    unittest.main()
