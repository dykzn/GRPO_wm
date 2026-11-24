"""Check process cleanup boundaries without signaling actual processes."""

import os
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stop_spatial_grpo import Process, protected_ancestors, ray_roots, select_processes, signal_process


RUN = Path("/tmp/example-grpo-run")
RAY_ROOT = "/tmp/spatial-grpo8-Ab1234"


def process(pid, ppid=1, name="python", argv=(), environment=None, uid=None, started=123):
    return Process(pid, ppid, os.geteuid() if uid is None else uid, started, name, tuple(argv), environment or {})


class SelectionTests(unittest.TestCase):
    def test_run_workers_and_descendants_survive_driver_exit(self):
        rows = {10: process(10, environment={"SPATIAL_GRPO_OUTPUT": str(RUN)}),
                11: process(11, 10),
                20: process(20, name="ray::ActorGroup", environment={"RAY_TMPDIR": RAY_ROOT}),
                30: process(30)}
        self.assertEqual(set(select_processes(rows, RUN, {RAY_ROOT}, set())), {10, 11, 20})

    def test_other_runs_other_users_and_callers_are_protected(self):
        rows = {10: process(10, environment={"SPATIAL_GRPO_OUTPUT": str(RUN)}),
                11: process(11, 10, environment={"SPATIAL_GRPO_OUTPUT": "/tmp/another-run"}),
                12: process(12, 10, uid=os.geteuid() + 1),
                13: process(13, 10),
                14: process(14, environment={"SPATIAL_GRPO_OUTPUT": str(RUN)})}
        self.assertEqual(set(select_processes(rows, RUN, set(), {13, 14})), {10})

    def test_ray_path_boundary_does_not_match_another_cluster_or_log_viewer(self):
        rows = {10: process(10, name="raylet", argv=[f"--session_dir={RAY_ROOT}/ray/session_2026_123"]),
                11: process(11, name="raylet", argv=[f"--session_dir={RAY_ROOT}extra/ray/session_2026_123"]),
                12: process(12, name="tail", argv=[f"{RAY_ROOT}/ray/session_2026_123/logs/worker.out"]),
                13: process(13, argv=["/venv/ray/dashboard/agent.py", f"--temp-dir={RAY_ROOT}"])}
        self.assertEqual(set(select_processes(rows, RUN, {RAY_ROOT}, set())), {10, 13})

    def test_generic_training_command_and_historical_numeric_pid_are_insufficient(self):
        rows = {1284539: process(1284539, argv=["python", "train_spatial_grpo.py"]),
                1293737: process(1293737, name="ray::RolloutGroup")}
        self.assertFalse(select_processes(rows, RUN, {RAY_ROOT}, set()))

    def test_tracked_identity_rejects_reused_pid(self):
        rows = {10: process(10, started=200), 11: process(11, started=100)}
        self.assertEqual(set(select_processes(rows, RUN, set(), set(), {10: 100, 11: 100})), {11})

    def test_ancestor_protection_handles_reparented_process_tree(self):
        rows = {10: process(10, 2), 2: process(2, 1), 1: process(1, 0)}
        self.assertEqual(protected_ancestors(rows, 10), {0, 1, 2, 10})

    def test_extracts_only_the_exact_private_ray_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train.log").write_text(f"Ray {RAY_ROOT}/ray/session_2026-10-19_15-31-14_123/logs\n")
            self.assertEqual(ray_roots(root), {RAY_ROOT})


class SignalTests(unittest.TestCase):
    @mock.patch("stop_spatial_grpo.os.kill")
    @mock.patch("stop_spatial_grpo.os.pidfd_open", return_value=99, create=True)
    @mock.patch("stop_spatial_grpo.os.close")
    @mock.patch("stop_spatial_grpo.signal.pidfd_send_signal", create=True)
    def test_pid_reuse_or_ownership_change_never_receives_signal(self, send, close, open_fd, kill):
        target = process(10, started=100)
        for current in [None, process(10, started=200), process(10, uid=os.geteuid() + 1, started=100)]:
            with mock.patch("stop_spatial_grpo.read_process", return_value=current):
                self.assertFalse(signal_process(target, signal.SIGTERM))
        send.assert_not_called()
        kill.assert_not_called()
        self.assertEqual(close.call_count, 3)

    @mock.patch("stop_spatial_grpo.os.kill")
    @mock.patch("stop_spatial_grpo.os.pidfd_open", return_value=99, create=True)
    @mock.patch("stop_spatial_grpo.os.close")
    @mock.patch("stop_spatial_grpo.signal.pidfd_send_signal", create=True)
    def test_verified_process_is_signaled_through_its_pidfd(self, send, close, open_fd, kill):
        target = process(10)
        with mock.patch("stop_spatial_grpo.read_process", return_value=target):
            self.assertTrue(signal_process(target, signal.SIGTERM))
        send.assert_called_once_with(99, signal.SIGTERM)
        kill.assert_not_called()
        close.assert_called_once_with(99)


if __name__ == "__main__":
    unittest.main()
