"""Email when one Spatial training run exits; run in the GPU host PID namespace."""

from __future__ import annotations

import argparse
import fcntl
import getpass
import hashlib
import json
import os
import smtplib
import socket
import ssl
import sys
import tempfile
import time
import warnings
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT.parent / ".config/grpo-notify/smtp.json"
_WARNING_TIMES: dict[str, float] = {}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")  # noqa: UP017 -- system Python 3.10


def watchdog_print(message: str) -> None:
    """A broken/full logging destination must not prevent monitoring or mail."""
    try:
        print(message, flush=True)
    except OSError:
        pass


def watchdog_warning(key: str, message: str) -> None:
    now = time.monotonic()
    if now - _WARNING_TIMES.get(key, float("-inf")) >= 60:
        _WARNING_TIMES[key] = now
        watchdog_print(message)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(descriptor, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(temporary, path)


@dataclass(frozen=True)
class MailSettings:
    host: str
    port: int
    security: str
    sender: str
    recipients: tuple[str, ...]
    username: str = ""
    password: str = field(default="", repr=False)
    auth_method: str = "auto"


MailConfig = Path | Callable[[], MailSettings]


def load_mail_settings(path: MailConfig) -> MailSettings:
    """Read local credentials without copying them to logs or watcher state."""
    if callable(path):
        return path()
    if not path.is_file():
        raise ValueError(
            f"Missing mail config: {path}. Run this script with --configure first."
        )
    config = json.loads(path.read_text())
    password = ""
    if config.get("password_file"):
        secret = Path(config["password_file"]).expanduser()
        if not secret.is_absolute():
            secret = path.parent / secret
        metadata = secret.stat()
        if metadata.st_uid != os.geteuid() or metadata.st_mode & 0o077:
            raise ValueError(
                f"SMTP password file must belong to you with mode 600: {secret}"
            )
        # A leading blank line from an editor is not part of the authorization code.
        # Preserve spaces, which may be intentional in a generic SMTP password.
        password = secret.read_text().strip("\r\n")
        if "\n" in password or "\r" in password:
            raise ValueError("SMTP password file must contain a single nonempty line")
    elif config.get("password_env"):
        password = os.environ.get(config["password_env"], "")
    recipients = config.get("to", [])
    if isinstance(recipients, str):
        recipients = [recipients]
    settings = MailSettings(
        host=str(config["host"]),
        port=int(config["port"]),
        security=config.get("security", "ssl"),
        sender=config["from"],
        recipients=tuple(recipients),
        username=config.get("username", ""),
        password=password,
        auth_method=str(config.get("auth_method", "auto")).lower(),
    )
    if (
        settings.security not in {"ssl", "starttls", "plain"}
        or not 1 <= settings.port <= 65535
    ):
        raise ValueError("Invalid SMTP security or port")
    if settings.auth_method not in {"auto", "login"}:
        raise ValueError("SMTP auth_method must be auto or login")
    if settings.auth_method == "login" and settings.security == "plain":
        raise ValueError("Explicit AUTH LOGIN requires ssl or starttls")
    for value in [settings.host, settings.sender, *settings.recipients]:
        if (
            not isinstance(value, str)
            or not value.strip()
            or "\n" in value
            or "\r" in value
        ):
            raise ValueError(
                "SMTP host and email addresses must be nonempty single lines"
            )
    if not settings.recipients:
        raise ValueError("At least one recipient is required")
    if settings.username and not settings.password:
        raise ValueError(
            "SMTP username is set but the password/authorization code is missing"
        )
    return settings


@contextmanager
def smtp_step(stage: str, progress=None):
    """Attach a controlled stage label without logging SMTP replies or AUTH data."""
    if progress is not None:
        progress(f"[SMTP] {stage} ...")
    try:
        yield
    except (OSError, smtplib.SMTPException) as error:
        error.smtp_stage = stage
        raise
    if progress is not None:
        progress(f"[SMTP] {stage} OK")


def mail_error_details(error: Exception) -> dict:
    """Only expose error types, our own stage labels and numeric status codes."""
    details = {"type": type(error).__name__}
    for key, attribute in (
        ("stage", "smtp_stage"),
        ("auth_method", "smtp_auth_method"),
        ("auth_step", "smtp_auth_step"),
    ):
        value = getattr(error, attribute, None)
        if value is not None:
            details[key] = value
    for key in ("smtp_code", "errno"):
        value = getattr(error, key, None)
        if isinstance(value, int):
            details[key] = value
    return details


def mail_error_summary(error: Exception) -> str:
    return ", ".join(
        f"{key}={value}" for key, value in mail_error_details(error).items()
    )


def persist_watch_state(path: Path, state: dict) -> bool:
    """Best-effort persistence; disk exhaustion must not block a stop email."""
    try:
        atomic_json(path, state)
    except OSError as error:
        watchdog_warning(
            f"state:{path}",
            f"Watchdog state write unavailable ({mail_error_summary(error)}); "
            "continuing detection and email delivery with state in memory.",
        )
        return False
    return True


@contextmanager
def smtp_connection(settings: MailSettings, progress=None):
    """Use the configured verified transport and identify its failing operation."""
    context = ssl.create_default_context()
    stage = (
        "connect_ssl_and_greeting"
        if settings.security == "ssl"
        else "connect_and_greeting"
    )
    with smtp_step(stage, progress):
        if settings.security == "ssl":
            client = smtplib.SMTP_SSL(
                settings.host, settings.port, timeout=15, context=context
            )
        else:
            client = smtplib.SMTP(settings.host, settings.port, timeout=15)
    try:
        with smtp_step("ehlo", progress):
            code, reply = client.ehlo()
            if code != 250:
                raise smtplib.SMTPHeloError(code, reply)
        if settings.security == "starttls":
            with smtp_step("starttls", progress):
                client.starttls(context=context)
            with smtp_step("ehlo_after_starttls", progress):
                code, reply = client.ehlo()
                if code != 250:
                    raise smtplib.SMTPHeloError(code, reply)
        if progress is not None:
            # Only print known protocol names; never dump the raw EHLO reply.
            advertised = set(client.esmtp_features.get("auth", "").upper().split())
            known = [
                name
                for name in ("CRAM-MD5", "PLAIN", "LOGIN", "XOAUTH2", "OAUTHBEARER")
                if name in advertised
            ]
            progress(f"[SMTP] advertised_auth={','.join(known) or 'none recognized'}")
        yield client
    finally:
        # Local cleanup must not turn an accepted message into a retry/duplicate.
        try:
            client.close()
        except OSError:
            pass


def authenticate_smtp(client, settings: MailSettings, progress=None) -> None:
    """Allow explicit challenge/response LOGIN instead of automatic negotiation."""
    phase = "auth_command"
    responses = 0

    def login_response(challenge):
        nonlocal phase, responses
        responses += 1
        if responses > 2:
            phase = "unexpected_challenge"
            raise smtplib.SMTPException("Unexpected extra AUTH LOGIN challenge")
        phase = "username_response" if responses == 1 else "authorization_response"
        if progress is not None:
            progress(f"[SMTP] login/{phase} ...")
        return settings.username if responses == 1 else settings.password

    with smtp_step("login", progress):
        if settings.auth_method == "auto":
            client.login(settings.username, settings.password)
            return
        if progress is not None:
            progress("[SMTP] auth_method=LOGIN; initial_response=off")
        try:
            advertised = client.esmtp_features.get("auth", "").upper().split()
            if "LOGIN" not in advertised:
                raise smtplib.SMTPNotSupportedError(
                    "Server did not advertise AUTH LOGIN"
                )
            client.auth("LOGIN", login_response, initial_response_ok=False)
        except (OSError, smtplib.SMTPException) as error:
            error.smtp_auth_method = "LOGIN"
            error.smtp_auth_step = phase
            raise


def send_mail(
    settings: MailSettings,
    subject: str,
    body: str,
    message_id: str,
    *,
    progress=None,
) -> None:
    """Submit through SMTP; return only after the relay accepts every recipient."""
    message = EmailMessage()
    message["From"] = settings.sender
    message["To"] = ", ".join(settings.recipients)
    message["Subject"] = subject
    message["Date"] = formatdate(localtime=False)
    message["Message-ID"] = message_id
    message.set_content(body)
    with smtp_connection(settings, progress) as client:
        if settings.username:
            authenticate_smtp(client, settings, progress)
        with smtp_step("submit_message", progress):
            refused = client.send_message(message)
            if refused:
                raise smtplib.SMTPRecipientsRefused(refused)


def smtp_progress(message: str) -> None:
    print(message, flush=True)


def diagnose_smtp(path: Path) -> None:
    """Check the configured connection and EHLO without logging in or sending."""
    settings = load_mail_settings(path)
    print(
        f"SMTP endpoint: {settings.host}:{settings.port}; security={settings.security}",
        flush=True,
    )
    with smtp_connection(settings, smtp_progress):
        pass
    print("SMTP connection/EHLO passed. No login or email was attempted.", flush=True)


def configure(path: Path, recipient: str | None) -> None:
    """Prompt in the user's terminal; store secrets in a private local file."""
    print(
        "Configure SMTP here. The authorization code is hidden and is not written to logs."
    )
    host = input("SMTP host: ").strip()
    security = input("Security [ssl/starttls/plain, default ssl]: ").strip() or "ssl"
    port = int(
        input(f"SMTP port [{465 if security == 'ssl' else 587}]: ").strip()
        or (465 if security == "ssl" else 587)
    )
    username = input("SMTP login (blank for a relay without authentication): ").strip()
    sender = input(f"Sender address [{username}]: ").strip() or username
    recipient = recipient or input("Recipient address: ").strip()
    config = {
        "host": host,
        "port": port,
        "security": security,
        "username": username,
        "from": sender,
        "to": [recipient],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    if username:
        password = getpass.getpass("SMTP password / authorization code: ")
        secret = path.parent / "smtp_password"
        descriptor = os.open(secret, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            stream.write(password)
        config["password_file"] = str(secret.resolve())
    atomic_json(path, config)
    test_email(path)


def update_auth_code(path: Path) -> None:
    """Replace the configured authorization-code file through hidden local input."""
    # Do not load the old secret: this action must also repair an empty/bad file.
    config = json.loads(path.read_text())
    username = config.get("username", "")
    if (
        not isinstance(username, str)
        or not username
        or any(c.isspace() for c in username)
    ):
        raise ValueError("Configure a valid SMTP username before updating its code")
    if not config.get("password_file"):
        raise ValueError(
            "This action requires password_file; use --configure to set one"
        )
    secret = Path(config["password_file"]).expanduser()
    if not secret.is_absolute():
        secret = path.parent / secret
    secret = secret.resolve()
    if secret.exists() and secret.stat().st_uid != os.geteuid():
        raise ValueError("SMTP authorization-code file must belong to you")
    print(f"Update the SMTP authorization code for {username}.", flush=True)
    with warnings.catch_warnings():
        # getpass otherwise falls back to visible input when no terminal exists.
        warnings.simplefilter("error", getpass.GetPassWarning)
        try:
            code = getpass.getpass("New SMTP authorization code (hidden): ").strip()
        except getpass.GetPassWarning:
            raise ValueError(
                "Run --update-auth-code in a terminal with hidden input"
            ) from None
    if not code or not code.isascii() or any(c.isspace() for c in code):
        raise ValueError("Authorization code must be one nonempty ASCII token")
    secret.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{secret.name}.", dir=secret.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(code)
        os.replace(temporary, secret)
    finally:
        Path(temporary).unlink(missing_ok=True)
    print("Authorization code updated (mode 600). Sending one test email.", flush=True)
    test_email(path)


def test_email(path: Path) -> None:
    settings = load_mail_settings(path)
    print(
        f"SMTP endpoint: {settings.host}:{settings.port}; security={settings.security}",
        flush=True,
    )
    send_mail(
        settings,
        "[GRPO] 邮件通知测试",
        f"邮件配置测试成功。\n主机：{socket.gethostname()}\n时间：{utc_now()}\n\n此邮件只验证发送通道；看门狗还需单独启动。",
        make_msgid(domain="training-watchdog.local"),
        progress=smtp_progress,
    )
    print("Test email accepted by SMTP. This does not start the watchdog.", flush=True)


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    start_ticks: int
    role: str


def process_stat(pid: int, proc_root: Path = Path("/proc")) -> tuple[int, str] | None:
    """Distinguish dead/zombie processes and PID reuse; permission errors propagate."""
    try:
        path = proc_root / str(pid)
        if path.stat().st_uid != os.geteuid():
            return None
        fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] in {"Z", "X"}:
            return None
        return int(fields[19]), fields[0]
    except (FileNotFoundError, ProcessLookupError):
        return None


def discover_processes(
    run: Path, proc_root: Path = Path("/proc")
) -> list[ProcessIdentity]:
    """Select only this run's launcher, learner, and Cosmos service processes."""
    found = []
    for entry in proc_root.iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            identity = process_stat(int(entry.name), proc_root)
            if identity is None:
                continue
            argv = [
                part.decode(errors="replace")
                for part in (entry / "cmdline").read_bytes().split(b"\0")
                if part
            ]
            role = None
            if (str(PROJECT.parent / "reproduction/launcher.py") in argv
                    or any(value.endswith("/spatial_wovr_launch.py") for value in argv)):
                role = "launcher"
            elif any(value.endswith("/train_spatial_wovr.py") for value in argv):
                role = "learner"
            elif "cosmos_bridge.cosmos_server" in argv:
                role = "cosmos"
            if role is None:
                continue
            output = None
            for value in (entry / "environ").read_bytes().split(b"\0"):
                if value.startswith(b"SPATIAL_WOVR_OUTPUT="):
                    output = value.split(b"=", 1)[1].decode()
            if output is None or Path(output).resolve() != run:
                continue
            if process_stat(int(entry.name), proc_root) == identity:
                found.append(ProcessIdentity(int(entry.name), identity[0], role))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return sorted(found, key=lambda item: item.pid)


def stop_event(
    run: Path, identities: list[ProcessIdentity], proc_root: Path = Path("/proc")
) -> dict | None:
    """Trust an exit marker or verified process death, never log inactivity."""
    marker = run / "exit_code"
    try:
        code = int(marker.read_text().strip())
    except (FileNotFoundError, ValueError):
        code = None
    if code is not None:
        return {
            "detected_utc": utc_now(),
            "exit_code": code,
            "reason": "Training exit_code recorded",
            "stopped_processes": [],
        }
    stopped = []
    for item in identities:
        current = process_stat(item.pid, proc_root)
        if current is None or current[0] != item.start_ticks:
            stopped.append(asdict(item))
    if stopped:
        return {
            "detected_utc": utc_now(),
            "exit_code": None,
            "reason": "Tracked process stopped before an exit code was recorded",
            "stopped_processes": stopped,
        }
    return None


def process_context() -> dict:
    return {
        "hostname": socket.gethostname(),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "pid_namespace": os.readlink("/proc/self/ns/pid"),
    }


@contextmanager
def exclusive_watch(path: Path):
    """Keep a single notifier for each run without touching training processes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("A watchdog already monitors this run") from error
        yield


def notification(run: Path, event: dict) -> tuple[str, str]:
    code = event["exit_code"]
    status = "已完成" if code == 0 else "异常退出" if code is not None else "进程已停止"
    stopped = (
        ", ".join(f"{p['role']} PID {p['pid']}" for p in event["stopped_processes"])
        or "见退出码"
    )
    body = (
        f"训练状态：{status}\n主机：{socket.gethostname()}\n检测时间：{event['detected_utc']}\n"
        f"运行目录：{run}\n退出码：{code if code is not None else '尚未记录'}\n"
        f"触发原因：{event['reason']}\n停止的进程：{stopped}\n训练日志：{run / 'train.log'}\n\n"
        "看门狗只负责通知，没有停止、重启或清理训练进程。\n"
    )
    return f"[GRPO {status}] {run.name}", body


def deliver_until_sent(run: Path, config: MailConfig, state_file: Path, state: dict) -> None:
    """Persist the stop event and retry delivery failures, reloading SMTP config."""
    state["status"] = "delivery_pending"
    state.setdefault(
        "message_id",
        f"<grpo-stop-{hashlib.sha256(str(run).encode()).hexdigest()}@training-watchdog.local>",
    )
    state.setdefault("mail_attempts", 0)
    persist_watch_state(state_file, state)
    subject, body = notification(run, state["event"])
    while True:
        state["mail_attempts"] += 1
        try:
            settings = load_mail_settings(config)
            send_mail(settings, subject, body, state["message_id"])
        except (OSError, ValueError, KeyError, smtplib.SMTPException) as error:
            # Record the error type, not server responses or credential values.
            state["last_mail_error_type"] = type(error).__name__
            state["last_mail_error"] = mail_error_details(error)
            state["updated_utc"] = utc_now()
            persist_watch_state(state_file, state)
            delay = min(60, 5 * 2 ** min(state["mail_attempts"] - 1, 4))
            watchdog_print(
                f"Email attempt {state['mail_attempts']} failed ({mail_error_summary(error)}); retry in {delay}s.",
            )
            time.sleep(delay)
            continue
        state.update(status="sent", sent_utc=utc_now())
        state.pop("last_mail_error_type", None)
        state.pop("last_mail_error", None)
        saved = persist_watch_state(state_file, state)
        watchdog_print(
            f"Stop notification accepted by SMTP. State saved: {saved}; {state_file}"
        )
        return


def watch(
    run: Path,
    config: MailConfig,
    state_dir: Path,
    interval: float,
    inspect_only: bool = False,
    *,
    failures_only: bool = False,
) -> None:
    if (
        not (run / "resolved_config.yaml").is_file()
        or not (run / "train.log").is_file()
    ):
        raise ValueError(f"Not an initialized training output: {run}")
    source_paths = run / "source_paths.json"
    expected_services = (
        int(json.loads(source_paths.read_text()).get("gpu_count", 8))
        if source_paths.is_file()
        else 8
    )
    if expected_services not in (4, 8):
        raise ValueError("Unsupported GPU count in training source_paths.json")
    identities = discover_processes(run)
    event = stop_event(run, identities)
    if inspect_only:
        print(
            json.dumps(
                {
                    "run_dir": str(run),
                    "processes": [asdict(p) for p in identities],
                    "stop_event": event,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        if not identities and event is None:
            raise ValueError(
                "No matching host processes visible. Run in the GPU terminal's PID namespace."
            )
        return
    load_mail_settings(config)
    with exclusive_watch(state_dir / "watchdog.lock"):
        state_file = state_dir / "state.json"
        state = json.loads(state_file.read_text()) if state_file.exists() else {}
        if state and state.get("run_dir") != str(run):
            raise ValueError("This --state-dir belongs to another training run")
        if state.get("status") == "sent":
            watchdog_print(f"This run was already notified: {state_file}")
            return
        if failures_only and state.get("status") == "completed":
            return
        if state.get("status") == "delivery_pending":
            deliver_until_sent(run, config, state_file, state)
            return
        context = process_context()
        if state and state.get("process_context") != context:
            raise ValueError(
                "Saved watcher belongs to another boot/PID namespace; use the original GPU terminal or a new --state-dir."
            )
        if state.get("tracked"):
            identities = [ProcessIdentity(**item) for item in state["tracked"]]
            event = stop_event(run, identities)
        if not identities and event is None:
            raise ValueError(
                "No matching launcher/learner visible. Run this watchdog in the same GPU terminal as training."
            )
        state.update(
            status="watching",
            run_dir=str(run),
            pid=os.getpid(),
            failures_only=failures_only,
            process_context=context,
            poll_interval_seconds=interval,
            expected_cosmos_services=expected_services,
            tracked=[asdict(p) for p in identities],
            updated_utc=utc_now(),
        )
        persist_watch_state(state_file, state)
        watchdog_print(
            f"WATCHDOG ARMED: {run}; interval={interval}s; tracking {len(identities)} processes; state={state_file}",
        )
        last_write = time.monotonic()
        while True:
            if event is not None:
                if failures_only and event["exit_code"] is None:
                    # The launcher polls the learner every two seconds, then writes exit_code.
                    time.sleep(10)
                    try:
                        event = stop_event(run, []) or event
                    except OSError as error:
                        watchdog_warning("exit_marker", f"Cannot recheck exit code ({mail_error_summary(error)}).")
                state["event"] = event
                if failures_only and event["exit_code"] == 0:
                    state.update(status="completed", updated_utc=utc_now())
                    persist_watch_state(state_file, state)
                else:
                    deliver_until_sent(run, config, state_file, state)
                return
            time.sleep(interval)
            try:
                event = stop_event(run, identities)
                # Include services/learner created after an early attachment.
                if event is None and (
                    sum(p.role == "cosmos" for p in identities) < expected_services
                    or not any(p.role == "learner" for p in identities)
                ):
                    known = {p.pid for p in identities}
                    identities.extend(
                        p for p in discover_processes(run) if p.pid not in known
                    )
            except OSError as error:
                watchdog_warning(
                    "process_check",
                    f"Process/filesystem check unavailable ({mail_error_summary(error)}); "
                    "retrying without declaring the job stopped.",
                )
            # A failed state write is independent of checking process liveness.
            # Advance the heartbeat schedule even on failure to avoid log floods.
            if time.monotonic() - last_write >= 30:
                state.update(
                    updated_utc=utc_now(), tracked=[asdict(p) for p in identities]
                )
                persist_watch_state(state_file, state)
                last_write = time.monotonic()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smtp-config", type=Path, default=DEFAULT_CONFIG)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--configure", action="store_true")
    action.add_argument("--test-email", action="store_true")
    action.add_argument(
        "--update-auth-code",
        action="store_true",
        help="Replace the existing authorization code using hidden input, then test email",
    )
    action.add_argument(
        "--diagnose-smtp",
        action="store_true",
        help="Check connection/TLS/greeting/EHLO without logging in or sending email",
    )
    action.add_argument("--run-dir", type=Path)
    action.add_argument("--latest-run-root", type=Path)
    parser.add_argument("--to", help="Recipient for interactive --configure")
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Inspect identities without sending mail or arming",
    )
    args = parser.parse_args()
    if not 0.1 <= args.interval <= 60:
        parser.error("--interval must be between 0.1 and 60 seconds")
    if args.configure:
        configure(args.smtp_config, args.to)
        return
    if args.test_email:
        test_email(args.smtp_config)
        return
    if args.update_auth_code:
        update_auth_code(args.smtp_config)
        return
    if args.diagnose_smtp:
        diagnose_smtp(args.smtp_config)
        return
    run = args.run_dir
    if run is None:
        choices = sorted(
            p
            for p in args.latest_run_root.iterdir()
            if p.is_dir() and (p / "resolved_config.yaml").is_file()
        )
        if not choices:
            parser.error("No initialized training run found")
        run = choices[-1]
    run = run.resolve()
    key = f"{run.name}_{hashlib.sha256(str(run).encode()).hexdigest()[:8]}"
    state_dir = args.state_dir or PROJECT / "runtime_artifacts/training_watchdog" / key
    watch(run, args.smtp_config, state_dir, args.interval, args.inspect)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, smtplib.SMTPException) as exc:
        # Avoid printing SMTP responses that might include sensitive material.
        print(
            f"Watchdog not active / notification not confirmed: {mail_error_summary(exc)}",
            file=sys.stderr,
        )
        if isinstance(exc, (ValueError, FileNotFoundError)):
            print(str(exc), file=sys.stderr)
        if isinstance(exc, smtplib.SMTPAuthenticationError) and exc.smtp_code == 535:
            print(
                "SMTP rejected authentication. Check this account's SMTP service and "
                "authorization code; use --update-auth-code to replace the code locally.",
                file=sys.stderr,
            )
        raise SystemExit(1) from None
