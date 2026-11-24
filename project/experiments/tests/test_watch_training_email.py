"""Exercise stop detection and email retries without contacting an email service."""

import base64
import contextlib
import errno
import io
import json
import os
import smtplib
import sys
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import watch_training_email as watchdog


def fake_process(root, pid, run, role="learner", start=100, state="S"):
    directory = root / str(pid)
    directory.mkdir(parents=True, exist_ok=True)
    fields = [state] + ["0"] * 18 + [str(start)] + ["0"] * 8
    (directory / "stat").write_text(
        f"{pid} (process with (spaces)) " + " ".join(fields)
    )
    argv = {
        "launcher": ["python", "/project/experiments/spatial_wovr_launch.py"],
        "learner": ["python", "/project/experiments/train_spatial_wovr.py"],
        "cosmos": ["python", "-m", "cosmos_bridge.cosmos_server"],
        "unrelated": ["python", "/another/project/run.py"],
    }[role]
    (directory / "cmdline").write_bytes(b"\0".join(x.encode() for x in argv))
    (directory / "environ").write_bytes(
        f"SPATIAL_WOVR_OUTPUT={run}\0HF_TOKEN=not-for-logs\0".encode()
    )
    return watchdog.ProcessIdentity(pid, start, role)


def make_run(directory):
    run = directory / "run"
    run.mkdir()
    (run / "resolved_config.yaml").write_text("runner: {}\n")
    (run / "train.log").write_text("A long-running world model can remain silent.\n")
    return run.resolve()


def settings():
    return watchdog.MailSettings(
        "smtp.example.org",
        465,
        "ssl",
        "sender@example.org",
        ("recipient@example.org",),
        "sender@example.org",
        "private-authorization-code",
    )


def make_mail_config(root, old_code="old-code"):
    secret = root / "authorization_code"
    secret.write_text(old_code)
    secret.chmod(0o600)
    config = root / "smtp.json"
    config.write_text(
        json.dumps(
            {
                "host": "smtp.example.org",
                "port": 465,
                "security": "ssl",
                "auth_method": "login",
                "from": "sender@example.org",
                "to": ["recipient@example.org"],
                "username": "sender@example.org",
                "password_file": "authorization_code",
            }
        )
    )
    return config, secret


class WatchdogTests(unittest.TestCase):
    def test_four_gpu_manifest_tracks_six_processes_without_waiting_for_eight_services(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = make_run(root)
            (run / "source_paths.json").write_text(json.dumps({"gpu_count": 4}))
            identities = [
                watchdog.ProcessIdentity(11, 100, "launcher"),
                watchdog.ProcessIdentity(12, 101, "learner"),
            ]
            identities.extend(
                watchdog.ProcessIdentity(20 + rank, 102 + rank, "cosmos")
                for rank in range(4)
            )
            event = {
                "detected_utc": watchdog.utc_now(),
                "exit_code": 1,
                "reason": "test exit",
                "stopped_processes": [],
            }
            with (
                patch.object(
                    watchdog, "discover_processes", return_value=identities
                ) as discover,
                patch.object(watchdog, "stop_event", side_effect=[None, None, event]),
                patch.object(watchdog, "load_mail_settings", return_value=settings()),
                patch.object(watchdog, "send_mail") as send,
                patch.object(watchdog.time, "sleep"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                watchdog.watch(run, root / "config", root / "state", 2)
            discover.assert_called_once()
            send.assert_called_once()
            saved = json.loads((root / "state/state.json").read_text())
            self.assertEqual(saved["expected_cosmos_services"], 4)
            self.assertEqual(len(saved["tracked"]), 6)

    def test_full_disk_does_not_block_stop_email_even_with_broken_log_pipe(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = make_run(root)
            (run / "exit_code").write_text("1")
            state = {"run_dir": str(run), "event": watchdog.stop_event(run, [])}
            with (
                patch.object(
                    watchdog,
                    "atomic_json",
                    side_effect=OSError(errno.ENOSPC, "private text"),
                ),
                patch.object(watchdog, "load_mail_settings", return_value=settings()),
                patch.object(watchdog, "send_mail") as send,
                patch("builtins.print", side_effect=BrokenPipeError),
            ):
                watchdog.deliver_until_sent(
                    run, root / "config", root / "state.json", state
                )
            send.assert_called_once()
            self.assertEqual(state["status"], "sent")
            self.assertEqual(state["mail_attempts"], 1)

    def test_full_disk_still_retries_failed_smtp_delivery_and_redacts_errors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = make_run(root)
            (run / "exit_code").write_text("1")
            state = {"run_dir": str(run), "event": watchdog.stop_event(run, [])}
            output = io.StringIO()
            with (
                patch.object(
                    watchdog,
                    "atomic_json",
                    side_effect=OSError(errno.ENOSPC, "private storage text"),
                ),
                patch.object(watchdog, "load_mail_settings", return_value=settings()),
                patch.object(
                    watchdog,
                    "send_mail",
                    side_effect=[
                        smtplib.SMTPServerDisconnected("private SMTP text"),
                        None,
                    ],
                ) as send,
                patch.object(watchdog.time, "sleep") as sleep,
                contextlib.redirect_stdout(output),
            ):
                watchdog.deliver_until_sent(
                    run, root / "config", root / "state.json", state
                )
            self.assertEqual(send.call_count, 2)
            sleep.assert_called_once_with(5)
            self.assertEqual(state["status"], "sent")
            self.assertNotIn("private storage text", output.getvalue())
            self.assertNotIn("private SMTP text", output.getvalue())
            self.assertNotIn(settings().password, output.getvalue())

    def test_watch_continues_detecting_exit_when_heartbeat_state_cannot_be_written(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = make_run(root)
            learner = watchdog.ProcessIdentity(100001, 100, "learner")
            event = {
                "detected_utc": watchdog.utc_now(),
                "exit_code": 1,
                "reason": "test exit",
                "stopped_processes": [],
            }
            ticks = iter(range(0, 10000, 31))
            with (
                patch.object(watchdog, "discover_processes", return_value=[learner]),
                patch.object(watchdog, "stop_event", side_effect=[None, None, event]),
                patch.object(watchdog, "load_mail_settings", return_value=settings()),
                patch.object(
                    watchdog,
                    "atomic_json",
                    side_effect=OSError(errno.ENOSPC, "private storage text"),
                ),
                patch.object(watchdog, "send_mail") as send,
                patch.object(watchdog.time, "sleep"),
                patch.object(
                    watchdog.time, "monotonic", side_effect=lambda: next(ticks)
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                watchdog.watch(run, root / "config", root / "state", 2)
            send.assert_called_once()
            self.assertIn("退出码：1", send.call_args.args[2])

    def test_discovery_uses_exact_run_and_excludes_unrelated_processes_and_secrets(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = make_run(root)
            proc = root / "proc"
            fake_process(proc, 11, run, "launcher")
            fake_process(proc, 12, run, "learner")
            fake_process(proc, 13, run, "cosmos")
            fake_process(proc, 14, root / "other", "learner")
            fake_process(proc, 15, run, "unrelated")
            identities = watchdog.discover_processes(run, proc)
            self.assertEqual([p.pid for p in identities], [11, 12, 13])
            self.assertNotIn("not-for-logs", repr(identities))
            self.assertNotIn("HF_TOKEN", json.dumps([asdict(p) for p in identities]))

    def test_stale_log_does_not_signal_failure_and_pid_reuse_or_zombie_does(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = make_run(root)
            proc = root / "proc"
            identity = fake_process(proc, 11, run)
            os.utime(run / "train.log", (1, 1))
            self.assertIsNone(watchdog.stop_event(run, [identity], proc))
            fake_process(proc, 11, run, start=101)
            event = watchdog.stop_event(run, [identity], proc)
            self.assertEqual(event["stopped_processes"], [asdict(identity)])
            fake_process(proc, 11, run, state="Z")
            self.assertIsNotNone(watchdog.stop_event(run, [identity], proc))

    def test_exit_marker_preserves_normal_completion_and_error_code(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = make_run(root)
            for code in [0, 1, -9, 130]:
                (run / "exit_code").write_text(str(code))
                event = watchdog.stop_event(run, [], root)
                self.assertEqual(event["exit_code"], code)
                subject, body = watchdog.notification(run, event)
                self.assertIn("已完成" if code == 0 else "异常退出", subject)
                self.assertIn(str(code), body)

    def test_inaccessible_process_state_does_not_become_a_false_stop(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = make_run(Path(temporary))
            with (
                patch.object(watchdog, "process_stat", side_effect=PermissionError),
                self.assertRaises(PermissionError),
            ):
                watchdog.stop_event(run, [watchdog.ProcessIdentity(11, 100, "learner")])

    def test_no_host_processes_visible_cannot_arm_or_send_a_false_alarm(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = make_run(root)
            with (
                patch.object(watchdog, "discover_processes", return_value=[]),
                patch.object(watchdog, "load_mail_settings", return_value=settings()),
                patch.object(watchdog, "send_mail") as send,
            ):
                with self.assertRaisesRegex(ValueError, "same GPU terminal"):
                    watchdog.watch(run, root / "config", root / "state", 2)
                send.assert_not_called()
                self.assertFalse((root / "state/state.json").exists())

    def test_duplicate_watcher_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            lock = Path(temporary) / "watchdog.lock"
            with (
                watchdog.exclusive_watch(lock),
                self.assertRaisesRegex(ValueError, "already monitors"),
                watchdog.exclusive_watch(lock),
            ):
                self.fail("Second watchdog obtained the same lock")

    def test_smtp_ssl_authenticates_and_sends_without_exposing_the_password(self):
        client = MagicMock()
        client.ehlo.return_value = (250, b"OK")
        client.send_message.return_value = {}
        with patch.object(watchdog.smtplib, "SMTP_SSL", return_value=client) as connect:
            watchdog.send_mail(
                settings(),
                "training stopped",
                "run status only",
                "<run@watchdog.local>",
            )
        connect.assert_called_once()
        client.login.assert_called_once_with(
            "sender@example.org", "private-authorization-code"
        )
        message = client.send_message.call_args.args[0]
        self.assertEqual(message["To"], "recipient@example.org")
        self.assertNotIn("private-authorization-code", message.as_string())
        self.assertNotIn("private-authorization-code", repr(settings()))
        client.close.assert_called_once()

    def test_starttls_is_used_before_authentication_and_refused_delivery_fails(self):
        client = MagicMock()
        client.ehlo.return_value = (250, b"OK")
        client.send_message.return_value = {"recipient@example.org": (550, b"refused")}
        config = watchdog.MailSettings(
            "smtp.example.org",
            587,
            "starttls",
            "sender@example.org",
            ("recipient@example.org",),
            "sender",
            "secret",
        )
        with (
            patch.object(watchdog.smtplib, "SMTP", return_value=client),
            self.assertRaises(smtplib.SMTPRecipientsRefused),
        ):
            watchdog.send_mail(config, "status", "body", "<run@watchdog.local>")
        methods = [call[0] for call in client.method_calls]
        self.assertLess(methods.index("starttls"), methods.index("login"))
        client.close.assert_called_once()

    def test_disconnect_stage_is_reported_without_server_text_or_credentials(self):
        for operation, stage in [
            ("connect", "connect_ssl_and_greeting"),
            ("ehlo", "ehlo"),
            ("login", "login"),
            ("send_message", "submit_message"),
        ]:
            with self.subTest(operation=operation):
                client = MagicMock()
                client.ehlo.return_value = (250, b"OK")
                client.send_message.return_value = {}
                error = smtplib.SMTPServerDisconnected(
                    "private-authorization-code echoed by a server"
                )
                if operation != "connect":
                    getattr(client, operation).side_effect = error
                output = io.StringIO()
                with (
                    patch.object(
                        watchdog.smtplib,
                        "SMTP_SSL",
                        side_effect=error if operation == "connect" else None,
                        return_value=client,
                    ),
                    contextlib.redirect_stdout(output),
                    self.assertRaises(smtplib.SMTPServerDisconnected) as caught,
                ):
                    watchdog.send_mail(
                        settings(),
                        "status",
                        "body",
                        "<run@watchdog.local>",
                        progress=watchdog.smtp_progress,
                    )
                self.assertIs(caught.exception, error)
                self.assertEqual(
                    watchdog.mail_error_details(error),
                    {"type": "SMTPServerDisconnected", "stage": stage},
                )
                self.assertIn(f"[SMTP] {stage} ...", output.getvalue())
                self.assertNotIn(f"[SMTP] {stage} OK", output.getvalue())
                safe_text = output.getvalue() + watchdog.mail_error_summary(error)
                self.assertNotIn("private-authorization-code", safe_text)
                self.assertNotIn("echoed by a server", safe_text)
                if operation != "connect":
                    client.close.assert_called_once()
                if operation in {"connect", "ehlo", "login"}:
                    client.send_message.assert_not_called()

    def test_smtp_rejection_reports_numeric_code_and_does_not_continue(self):
        client = MagicMock()
        client.ehlo.return_value = (550, b"private-authorization-code")
        with (
            patch.object(watchdog.smtplib, "SMTP_SSL", return_value=client),
            self.assertRaises(smtplib.SMTPHeloError) as caught,
        ):
            watchdog.send_mail(settings(), "status", "body", "<run@watchdog.local>")
        self.assertEqual(
            watchdog.mail_error_details(caught.exception),
            {"type": "SMTPHeloError", "stage": "ehlo", "smtp_code": 550},
        )
        client.login.assert_not_called()
        client.send_message.assert_not_called()
        client.close.assert_called_once()

    def test_starttls_failure_never_attempts_login_or_email(self):
        client = MagicMock()
        client.ehlo.return_value = (250, b"OK")
        client.starttls.side_effect = smtplib.SMTPServerDisconnected("private reply")
        config = watchdog.MailSettings(
            "smtp.example.org",
            587,
            "starttls",
            "sender@example.org",
            ("recipient@example.org",),
            "sender",
            "secret",
        )
        with (
            patch.object(watchdog.smtplib, "SMTP", return_value=client),
            self.assertRaises(smtplib.SMTPServerDisconnected) as caught,
        ):
            watchdog.send_mail(config, "status", "body", "<run@watchdog.local>")
        self.assertEqual(
            watchdog.mail_error_details(caught.exception)["stage"], "starttls"
        )
        client.login.assert_not_called()
        client.send_message.assert_not_called()
        client.close.assert_called_once()

    def test_cleanup_failure_does_not_resubmit_an_already_accepted_email(self):
        client = MagicMock()
        client.ehlo.return_value = (250, b"OK")
        client.send_message.return_value = {}
        client.close.side_effect = OSError("private reply")
        with patch.object(watchdog.smtplib, "SMTP_SSL", return_value=client):
            watchdog.send_mail(settings(), "status", "body", "<run@watchdog.local>")
        client.send_message.assert_called_once()

    def test_connection_diagnosis_never_sends_credentials_or_email(self):
        client = MagicMock()
        client.ehlo.return_value = (250, b"OK")
        output = io.StringIO()
        with (
            patch.object(watchdog, "load_mail_settings", return_value=settings()),
            patch.object(watchdog.smtplib, "SMTP_SSL", return_value=client),
            contextlib.redirect_stdout(output),
        ):
            watchdog.diagnose_smtp(Path("config"))
        client.ehlo.assert_called_once()
        client.login.assert_not_called()
        client.send_message.assert_not_called()
        client.close.assert_called_once()
        self.assertIn("No login or email was attempted", output.getvalue())
        self.assertNotIn("private-authorization-code", output.getvalue())

    def test_explicit_login_uses_two_challenges_without_initial_credentials(self):
        config = replace(settings(), auth_method="login")
        client = smtplib.SMTP(local_hostname="watchdog.test")
        client.esmtp_features = {"auth": "CRAM-MD5 PLAIN LOGIN"}
        output = io.StringIO()
        with (
            patch.object(
                client,
                "docmd",
                side_effect=[
                    (334, base64.b64encode(b"Username:")),
                    (334, base64.b64encode(b"Password:")),
                    (235, b"Accepted"),
                ],
            ) as command,
            patch.object(client, "login") as automatic,
            contextlib.redirect_stdout(output),
        ):
            watchdog.authenticate_smtp(client, config, watchdog.smtp_progress)
        automatic.assert_not_called()
        self.assertEqual(command.call_args_list[0].args, ("AUTH", "LOGIN"))
        self.assertEqual(
            base64.b64decode(command.call_args_list[1].args[0]).decode(),
            config.username,
        )
        self.assertEqual(
            base64.b64decode(command.call_args_list[2].args[0]).decode(),
            config.password,
        )
        self.assertIn("auth_method=LOGIN; initial_response=off", output.getvalue())
        self.assertIn("[SMTP] login OK", output.getvalue())
        self.assertNotIn(config.password, output.getvalue())
        self.assertNotIn(
            base64.b64encode(config.password.encode()).decode(), output.getvalue()
        )

    def test_explicit_login_disconnect_identifies_last_response_without_retry(self):
        for index, phase in enumerate(
            ("auth_command", "username_response", "authorization_response")
        ):
            with self.subTest(phase=phase):
                client = smtplib.SMTP(local_hostname="watchdog.test")
                client.esmtp_features = {"auth": "PLAIN LOGIN"}
                replies = [
                    (334, base64.b64encode(b"Username:")),
                    (334, base64.b64encode(b"Password:")),
                ][:index]
                replies.append(smtplib.SMTPServerDisconnected("private server reply"))
                output = io.StringIO()
                with (
                    patch.object(client, "docmd", side_effect=replies) as command,
                    contextlib.redirect_stdout(output),
                    self.assertRaises(smtplib.SMTPServerDisconnected) as caught,
                ):
                    watchdog.authenticate_smtp(
                        client,
                        replace(settings(), auth_method="login"),
                        watchdog.smtp_progress,
                    )
                self.assertEqual(command.call_count, index + 1)
                self.assertEqual(
                    watchdog.mail_error_details(caught.exception),
                    {
                        "type": "SMTPServerDisconnected",
                        "stage": "login",
                        "auth_method": "LOGIN",
                        "auth_step": phase,
                    },
                )
                safe_text = output.getvalue() + watchdog.mail_error_summary(
                    caught.exception
                )
                self.assertNotIn("private server reply", safe_text)
                self.assertNotIn(settings().password, safe_text)

    def test_explicit_login_requires_advertised_support_and_rejects_extra_challenges(
        self,
    ):
        client = smtplib.SMTP(local_hostname="watchdog.test")
        client.esmtp_features = {"auth": "PLAIN"}
        config = replace(settings(), auth_method="login")
        with (
            patch.object(client, "docmd") as command,
            self.assertRaises(smtplib.SMTPNotSupportedError),
        ):
            watchdog.authenticate_smtp(client, config)
        command.assert_not_called()
        client.esmtp_features = {"auth": "LOGIN"}
        with (
            patch.object(client, "docmd", return_value=(334, b"")) as command,
            self.assertRaises(smtplib.SMTPException) as caught,
        ):
            watchdog.authenticate_smtp(client, config)
        self.assertEqual(command.call_count, 3)
        self.assertEqual(caught.exception.smtp_auth_step, "unexpected_challenge")

    def test_auth_method_config_preserves_auto_default_and_requires_tls_for_login(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            secret = root / "password"
            secret.write_text("secret")
            secret.chmod(0o600)
            config_file = root / "smtp.json"
            config = {
                "host": "smtp.example.org",
                "port": 465,
                "security": "ssl",
                "from": "sender@example.org",
                "to": ["recipient@example.org"],
                "username": "sender@example.org",
                "password_file": str(secret),
            }
            config_file.write_text(json.dumps(config))
            self.assertEqual(
                watchdog.load_mail_settings(config_file).auth_method, "auto"
            )
            config["auth_method"] = "login"
            config_file.write_text(json.dumps(config))
            self.assertEqual(
                watchdog.load_mail_settings(config_file).auth_method, "login"
            )
            config["security"] = "plain"
            config_file.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "requires ssl or starttls"):
                watchdog.load_mail_settings(config_file)

    def test_mail_test_shows_progress_without_arming_training_monitor(self):
        client = MagicMock()
        client.ehlo.return_value = (250, b"OK")
        client.send_message.return_value = {}
        output = io.StringIO()
        with (
            patch.object(watchdog, "load_mail_settings", return_value=settings()),
            patch.object(watchdog.smtplib, "SMTP_SSL", return_value=client),
            patch.object(watchdog, "watch") as watch,
            contextlib.redirect_stdout(output),
        ):
            watchdog.test_email(Path("config"))
        watch.assert_not_called()
        self.assertIn("[SMTP] login OK", output.getvalue())
        self.assertIn("[SMTP] submit_message OK", output.getvalue())
        self.assertIn("Test email accepted by SMTP", output.getvalue())
        self.assertNotIn("private-authorization-code", output.getvalue())

    def test_failed_delivery_is_persisted_retried_and_marked_sent_after_acceptance(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = make_run(root)
            (run / "exit_code").write_text("1")
            state = {"run_dir": str(run), "event": watchdog.stop_event(run, [])}
            state_file = root / "state/state.json"
            with (
                patch.object(
                    watchdog, "load_mail_settings", return_value=settings()
                ) as load,
                patch.object(
                    watchdog,
                    "send_mail",
                    side_effect=[
                        smtplib.SMTPServerDisconnected("do not copy server text"),
                        None,
                    ],
                ) as send,
                patch.object(watchdog.time, "sleep") as sleep,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                watchdog.deliver_until_sent(run, root / "config", state_file, state)
            saved = json.loads(state_file.read_text())
            self.assertEqual(saved["status"], "sent")
            self.assertEqual(saved["mail_attempts"], 2)
            self.assertEqual(load.call_count, 2)
            self.assertEqual(
                send.call_args_list[0].args[-1], send.call_args_list[1].args[-1]
            )
            sleep.assert_called_once_with(5)
            self.assertNotIn("private-authorization-code", state_file.read_text())
            self.assertNotIn("do not copy server text", state_file.read_text())

    def test_sent_state_prevents_resending_on_watchdog_restart(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = make_run(root)
            state_dir = root / "state"
            watchdog.atomic_json(
                state_dir / "state.json", {"status": "sent", "run_dir": str(run)}
            )
            with (
                patch.object(watchdog, "discover_processes", return_value=[]),
                patch.object(watchdog, "load_mail_settings", return_value=settings()),
                patch.object(watchdog, "send_mail") as send,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                watchdog.watch(run, root / "config", state_dir, 2)
            send.assert_not_called()

    def test_watch_attaches_new_learner_then_notifies_its_disappearance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = make_run(root)
            launcher = watchdog.ProcessIdentity(100001, 100, "launcher")
            learner = watchdog.ProcessIdentity(100002, 101, "learner")
            event = {
                "detected_utc": watchdog.utc_now(),
                "exit_code": None,
                "reason": "Tracked learner stopped",
                "stopped_processes": [asdict(learner)],
            }
            with (
                patch.object(
                    watchdog,
                    "discover_processes",
                    side_effect=[[launcher], [launcher, learner]],
                ) as discover,
                patch.object(watchdog, "stop_event", side_effect=[None, None, event]),
                patch.object(watchdog, "load_mail_settings", return_value=settings()),
                patch.object(watchdog, "send_mail") as send,
                patch.object(watchdog.time, "sleep"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                watchdog.watch(run, root / "config", root / "state", 2)
            self.assertEqual(discover.call_count, 2)
            send.assert_called_once()
            self.assertIn("learner PID 100002", send.call_args.args[2])
            state = json.loads((root / "state/state.json").read_text())
            self.assertEqual(state["status"], "sent")
            self.assertIsNone(state["event"]["exit_code"])

    def test_other_pid_namespace_cannot_resume_watching_and_send_a_false_alarm(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = make_run(root)
            state_dir = root / "state"
            watchdog.atomic_json(
                state_dir / "state.json",
                {
                    "status": "watching",
                    "run_dir": str(run),
                    "process_context": {"pid_namespace": "another namespace"},
                },
            )
            with (
                patch.object(watchdog, "discover_processes", return_value=[]),
                patch.object(watchdog, "load_mail_settings", return_value=settings()),
                patch.object(watchdog, "send_mail") as send,
                self.assertRaisesRegex(ValueError, "another boot/PID namespace"),
            ):
                watchdog.watch(run, root / "config", state_dir, 2)
            send.assert_not_called()

    def test_interactive_setup_keeps_authorization_code_in_a_private_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "smtp.json"
            answers = [
                "smtp.example.org",
                "ssl",
                "465",
                "sender@example.org",
                "",
                "recipient@example.org",
            ]
            with (
                patch("builtins.input", side_effect=answers),
                patch.object(
                    watchdog.getpass, "getpass", return_value="authorization-code"
                ),
                patch.object(watchdog, "test_email") as test,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                watchdog.configure(config, None)
            test.assert_called_once_with(config)
            self.assertEqual(config.stat().st_mode & 0o777, 0o600)
            secret = Path(json.loads(config.read_text())["password_file"])
            self.assertEqual(secret.stat().st_mode & 0o777, 0o600)
            self.assertNotIn("authorization-code", config.read_text())
            self.assertEqual(
                watchdog.load_mail_settings(config).password, "authorization-code"
            )

    def test_world_readable_secret_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            secret = root / "password"
            secret.write_text("secret")
            secret.chmod(0o644)
            config = root / "smtp.json"
            config.write_text(json.dumps({"password_file": str(secret)}))
            with self.assertRaisesRegex(ValueError, "mode 600"):
                watchdog.load_mail_settings(config)

    def test_auth_code_update_repairs_empty_secret_and_preserves_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            config, secret = make_mail_config(Path(temporary), old_code="\n\n")
            original_config = config.read_bytes()
            output = io.StringIO()
            with (
                patch.object(
                    watchdog.getpass, "getpass", return_value=" new-auth-code "
                ),
                patch.object(watchdog, "test_email") as test,
                contextlib.redirect_stdout(output),
            ):
                watchdog.update_auth_code(config)
            test.assert_called_once_with(config)
            self.assertEqual(config.read_bytes(), original_config)
            self.assertEqual(secret.read_text(), "new-auth-code")
            self.assertEqual(secret.stat().st_mode & 0o777, 0o600)
            self.assertEqual(
                watchdog.load_mail_settings(config).password, "new-auth-code"
            )
            self.assertNotIn("new-auth-code", output.getvalue())
            self.assertIn("Sending one test email", output.getvalue())

    def test_invalid_or_visible_auth_code_input_keeps_old_secret_and_never_sends(self):
        for new_code in ["", "\n\n", "code with spaces", "first\nsecond", "非ASCII"]:
            with (
                self.subTest(input_format=new_code),
                tempfile.TemporaryDirectory() as tmp,
            ):
                config, secret = make_mail_config(Path(tmp))
                with (
                    patch.object(watchdog.getpass, "getpass", return_value=new_code),
                    patch.object(watchdog, "test_email") as test,
                    contextlib.redirect_stdout(io.StringIO()),
                    self.assertRaisesRegex(ValueError, "nonempty ASCII token"),
                ):
                    watchdog.update_auth_code(config)
                self.assertEqual(secret.read_text(), "old-code")
                test.assert_not_called()
        with tempfile.TemporaryDirectory() as tmp:
            config, secret = make_mail_config(Path(tmp))
            with (
                patch.object(
                    watchdog.getpass,
                    "getpass",
                    side_effect=watchdog.getpass.GetPassWarning,
                ),
                patch.object(watchdog, "test_email") as test,
                contextlib.redirect_stdout(io.StringIO()),
                self.assertRaisesRegex(ValueError, "terminal with hidden input"),
            ):
                watchdog.update_auth_code(config)
            self.assertEqual(secret.read_text(), "old-code")
            test.assert_not_called()

    def test_failed_auth_code_replacement_preserves_old_file_and_removes_temporary(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, secret = make_mail_config(root)
            with (
                patch.object(watchdog.getpass, "getpass", return_value="new-auth-code"),
                patch.object(watchdog.os, "replace", side_effect=PermissionError),
                patch.object(watchdog, "test_email") as test,
                contextlib.redirect_stdout(io.StringIO()),
                self.assertRaises(PermissionError),
            ):
                watchdog.update_auth_code(config)
            self.assertEqual(secret.read_text(), "old-code")
            self.assertEqual(
                sorted(p.name for p in root.iterdir()),
                ["authorization_code", "smtp.json"],
            )
            test.assert_not_called()

    def test_secret_file_ignores_boundary_blank_lines_but_preserves_password_spaces(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            secret = root / "password"
            secret.touch(mode=0o600)
            config = root / "smtp.json"
            config.write_text(
                json.dumps(
                    {
                        "host": "smtp.example.org",
                        "port": 465,
                        "security": "ssl",
                        "from": "sender@example.org",
                        "to": ["recipient@example.org"],
                        "username": "sender@example.org",
                        "password_file": str(secret),
                    }
                )
            )
            for raw, expected in [
                ("\nauthorization-code\n", "authorization-code"),
                ("\r\n\r\nauthorization-code\r\n\r\n", "authorization-code"),
                ("\n password with spaces \n", " password with spaces "),
            ]:
                secret.write_text(raw)
                self.assertEqual(watchdog.load_mail_settings(config).password, expected)
            secret.write_text("first-authorization-code\nsecond-authorization-code")
            with self.assertRaisesRegex(ValueError, "single nonempty line") as caught:
                watchdog.load_mail_settings(config)
            self.assertNotIn("first-authorization-code", str(caught.exception))
            self.assertNotIn("second-authorization-code", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
