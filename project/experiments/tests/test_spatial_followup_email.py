"""Outcome notifications must survive workload failures and SMTP interruptions."""

from __future__ import annotations

import contextlib
import errno
import fcntl
import io
import json
import os
import smtplib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

EXPERIMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS))

import spatial_followup_email as notify
from stop_spatial_grpo import read_process


class OutcomeMailTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.plan = {
            "output": str(self.root),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        }
        self.config = self.root / "not-a-real-smtp-config.json"
        self.output = io.StringIO()
        self.redirect = contextlib.redirect_stdout(self.output)
        self.redirect.__enter__()
        self.addCleanup(self.redirect.__exit__, None, None, None)

    def write(self, name, value):
        (self.root / name).write_text(json.dumps(value))

    def mailer(self):
        return notify.OutcomeMailer(self.plan, self.config)

    def test_pending_or_never_started_watchdogs_do_not_send(self):
        self.write(
            "primary_state.json", {"status": "running", "stage": "world_model_update"}
        )
        self.write(
            "primary_identity.json",
            {"pid": os.getpid(), "start_ticks": read_process(os.getpid()).started},
        )
        mailer = self.mailer()
        with patch.object(notify, "send_mail") as sender:
            mailer.scan()
            mailer.send_due()
        sender.assert_not_called()
        self.assertEqual(mailer.events, {})
        self.assertFalse(mailer.complete())

    def test_wm_failure_and_fallback_failure_both_queue_actionable_mail(self):
        self.write("primary_state.json", {"status": "cleaning_failed_workload"})
        self.write("primary_failed.json", {"error": "WM update probe exited 1"})
        (self.root / "primary_events.jsonl").write_text(
            json.dumps(
                {
                    "status": "running",
                    "stage": "world_model_update",
                    "log": "/logs/update.log",
                }
            )
            + "\n"
            + '{"incomplete_event":'
        )
        self.write(
            "fallback_state.json", {"status": "failed", "error": "GRPO exited 1"}
        )
        mailer = self.mailer()
        mailer.scan()
        self.assertEqual({e["outcome"] for e in mailer.events.values()}, {"failed"})
        self.assertEqual(len(mailer.events), 2)
        body = mailer.events["primary-outcome"]["body"]
        self.assertIn("WM update probe exited 1", body)
        self.assertIn("world_model_update", body)
        self.assertIn("/logs/update.log", body)
        self.assertFalse(mailer.complete())

    def test_success_also_notifies_backup_not_needed(self):
        self.write("primary_done.json", {"completed_utc": "now"})
        self.write("fallback_state.json", {"status": "not_needed"})
        mailer = self.mailer()
        mailer.scan()
        self.assertEqual(mailer.events["primary-outcome"]["outcome"], "complete")
        self.assertEqual(mailer.events["fallback-outcome"]["outcome"], "not_needed")
        with (
            patch.object(notify, "load_mail_settings"),
            patch.object(notify, "send_mail") as sender,
        ):
            mailer.send_due()
        self.assertEqual(sender.call_count, 2)
        self.assertTrue(mailer.complete())

    def test_real_process_exit_is_detected_for_either_role(self):
        child = subprocess.Popen(
            [sys.executable, "-c", "import sys; sys.stdin.read()"],
            stdin=subprocess.PIPE,
        )
        try:
            current = read_process(child.pid)
            for role in notify.ROLES:
                self.write(f"{role}_state.json", {"status": "running"})
                self.write(
                    f"{role}_identity.json",
                    {"pid": current.pid, "start_ticks": current.started},
                )
                self.assertIsNone(notify.observe_outcome(self.plan, role))
            child.communicate(timeout=10)
            for role in notify.ROLES:
                result = notify.observe_outcome(self.plan, role)
                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["evidence"], "supervisor_exit")
        finally:
            if child.poll() is None:
                child.terminate()
                child.communicate(timeout=10)

    def test_pid_reuse_cannot_hide_a_dead_watchdog(self):
        current = read_process(os.getpid())
        self.write(
            "fallback_identity.json",
            {"pid": current.pid, "start_ticks": current.started + 1},
        )
        self.write("fallback_state.json", {"status": "standby"})
        self.assertEqual(
            notify.observe_outcome(self.plan, "fallback")["status"], "failed"
        )

    def test_terminal_state_is_reread_after_death_to_avoid_false_failure(self):
        self.write("primary_identity.json", {"pid": 123, "start_ticks": 456})
        self.write("primary_state.json", {"status": "running"})

        def finished_during_check(identity):
            self.write("primary_state.json", {"status": "complete"})
            return False

        with patch.object(notify, "process_alive", side_effect=finished_during_check):
            self.assertEqual(
                notify.observe_outcome(self.plan, "primary")["status"], "complete"
            )

    def test_completed_grpo_is_recovered_when_supervisor_dies_before_marker(self):
        self.write("primary_identity.json", {"pid": 123, "start_ticks": 456})
        self.write(
            "primary_state.json", {"status": "stage_complete", "target_step": 40}
        )
        with (
            patch.object(notify, "process_alive", return_value=False),
            patch(
                "spatial_followup_watchdogs.finished_grpo", return_value=True
            ) as finished,
        ):
            result = notify.observe_outcome(self.plan, "primary")
        self.assertEqual(result["status"], "complete")
        finished.assert_called_once_with(self.root / "grpo_primary", 40)

    def test_failed_smtp_retries_after_restart_without_blocking_other_result(self):
        for role in notify.ROLES:
            self.write(f"{role}_state.json", {"status": "failed"})
        mailer = self.mailer()
        mailer.scan()
        error = smtplib.SMTPAuthenticationError(
            535, b"authorization-secret-must-not-be-logged"
        )
        settings = object()
        with (
            patch.object(notify, "load_mail_settings", return_value=settings),
            patch.object(notify, "send_mail", side_effect=[error, None]) as sender,
        ):
            mailer.send_due(now=100)
        self.assertEqual(sender.call_count, 2)
        first_id = sender.call_args_list[0].args[3]
        self.assertEqual(mailer.deliveries["primary-outcome"]["status"], "retrying")
        self.assertEqual(mailer.deliveries["fallback-outcome"]["status"], "sent")
        serialized = json.dumps(mailer.status()) + self.output.getvalue()
        self.assertNotIn("authorization-secret", serialized)
        self.assertIn("535", serialized)
        self.assertFalse(mailer.complete())

        restarted = self.mailer()
        with (
            patch.object(notify, "load_mail_settings", return_value=settings) as load,
            patch.object(notify, "send_mail") as sender,
        ):
            restarted.send_due(now=129)
            sender.assert_not_called()
            restarted.send_due(now=131)
            sender.assert_called_once()
            self.assertEqual(sender.call_args.args[3], first_id)
            load.assert_called_once_with(self.config)
        self.assertTrue(restarted.complete())
        again = self.mailer()
        again.scan()
        with patch.object(notify, "send_mail") as sender:
            again.send_due(now=1000)
        sender.assert_not_called()
        self.assertEqual(len(again.events), 2)

    def test_disk_full_keeps_attempting_mail_in_memory(self):
        for role in notify.ROLES:
            self.write(f"{role}_state.json", {"status": "failed"})
        mailer = self.mailer()
        with (
            patch.object(
                notify, "atomic_json", side_effect=OSError(errno.ENOSPC, "disk full")
            ),
            patch.object(notify, "load_mail_settings"),
            patch.object(notify, "send_mail") as sender,
        ):
            mailer.scan()
            mailer.send_due()
        self.assertEqual(sender.call_count, 2)
        self.assertTrue(mailer.complete())

    def test_cancellation_notifies_both_without_claiming_success(self):
        self.write("disarmed.json", {"cancelled_utc": "now"})
        mailer = self.mailer()
        mailer.scan()
        self.assertEqual(len(mailer.events), 2)
        self.assertEqual({e["outcome"] for e in mailer.events.values()}, {"cancelled"})
        self.assertFalse(mailer.complete())

    def test_activation_is_labelled_test_and_queued_once(self):
        mailer = self.mailer()
        mailer.confirm_armed()
        mailer.confirm_armed()
        self.assertEqual(len(mailer.events), 1)
        self.assertIn("测试", mailer.events["armed"]["subject"])
        self.assertFalse(mailer.complete())

    def test_duplicate_sender_is_rejected_before_sending(self):
        self.write("plan.json", self.plan)
        root = self.root / "outcome_email"
        root.mkdir()
        with (root / "notifier.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with (
                patch.object(
                    sys,
                    "argv",
                    ["notify", "--plan", str(self.root / "plan.json"), "--once"],
                ),
                patch.object(notify, "send_mail") as sender,
            ):
                with self.assertRaises(BlockingIOError):
                    notify.main()
                sender.assert_not_called()


if __name__ == "__main__":
    unittest.main()
