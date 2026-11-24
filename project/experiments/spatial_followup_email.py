"""Email the terminal outcomes of both ordered watchdogs, independently of training."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import smtplib
import socket
import time
from datetime import UTC, datetime
from pathlib import Path

from queue_spatial_wm_update import process_alive
from watch_training_email import (
    DEFAULT_CONFIG,
    atomic_json,
    load_mail_settings,
    mail_error_details,
    send_mail,
    watchdog_print,
    watchdog_warning,
)

ROLES = ("primary", "fallback")
TERMINAL = {"complete", "failed", "not_needed", "cancelled"}
ROLE_LABELS = {"primary": "主链", "fallback": "备用链"}
STATUS_LABELS = {
    "complete": "成功",
    "failed": "失败",
    "not_needed": "无需启动（主链成功）",
    "cancelled": "已撤销",
}


def utc() -> str:
    return datetime.now(UTC).isoformat()


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}


def recorded_outcome(output: Path, role: str) -> dict | None:
    details = read_json(output / f"{role}_state.json")
    if details.get("status") in TERMINAL:
        return details
    if (output / f"{role}_done.json").exists():
        return {**details, "status": "complete", "evidence": "completion_marker"}
    cancelled = (output / "disarmed.json").exists()
    if role == "primary" and (output / "primary_failed.json").exists():
        return {
            **details,
            **read_json(output / "primary_failed.json"),
            "status": "cancelled" if cancelled else "failed",
            "evidence": "failure_marker",
        }
    if cancelled:
        return {**details, "status": "cancelled", "evidence": "disarmed_marker"}
    return None


def observe_outcome(plan: dict, role: str) -> dict | None:
    output = Path(plan["output"])
    result = recorded_outcome(output, role)
    if result is not None:
        return result
    identity = read_json(output / f"{role}_identity.json")
    # A prepared role that has never started is not a failed process.
    if not identity or process_alive(identity):
        return None
    # Re-read after confirming death: the process may have finished between
    # the first state read and the PID/start-time check.
    result = recorded_outcome(output, role)
    if result is not None:
        return result
    details = read_json(output / f"{role}_state.json")
    if details.get("target_step"):
        from spatial_followup_watchdogs import finished_grpo

        if finished_grpo(output / f"grpo_{role}", details["target_step"]):
            return {
                **details,
                "status": "complete",
                "evidence": "completed_grpo_after_supervisor_exit",
            }
    return {
        **details,
        "status": "failed",
        "evidence": "supervisor_exit",
        "error": "Watchdog exited without a terminal result; PID/start time no longer matches.",
        "identity": identity,
    }


def outcome_event(plan: dict, role: str, result: dict) -> dict:
    output = Path(plan["output"])
    context = {}
    events = output / f"{role}_events.jsonl"
    if events.exists():
        for line in events.read_text().splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue  # A supervisor can die while appending its last line.
            for key in (
                "stage",
                "log",
                "resume",
                "checkpoint",
                "target_step",
                "grpo_output",
            ):
                if key in event:
                    context[key] = event[key]
    context.update(result)
    status = result["status"]
    label = f"{ROLE_LABELS[role]}{STATUS_LABELS[status]}"
    lines = [
        f"Spatial 顺序看门狗：{label}",
        f"节点：{socket.gethostname()}",
        f"观察时间（UTC）：{utc()}",
        f"计划目录：{output}",
        f"结果依据：{result.get('evidence', 'watchdog_state')}",
    ]
    for key, name in (
        ("stage", "最后阶段"),
        ("target_step", "目标 GRPO 步数"),
        ("resume", "续训断点"),
        ("checkpoint", "选择的断点"),
        ("grpo_output", "GRPO 输出"),
        ("log", "阶段日志"),
        ("error", "错误"),
    ):
        if key in context:
            lines.append(f"{name}：{str(context[key])[:2000]}")
    if role == "primary":
        lines.append(f"模型对：{plan.get('pair_manifest', '')}")
        if status == "failed":
            lines.append(
                "备用路径配置为使用原 world model 接手；当前进度请查看 fallback_state.json。"
            )
        wm = read_json(output / "world_model_update/state.json")
        if wm:
            lines.append(f"World model 更新状态：{json.dumps(wm, ensure_ascii=False)}")
    elif status == "complete":
        lines.append(
            "备用训练使用原 Wan/Cosmos，已完成额外 20 步并检查完整续训状态。"
        )
    elif status == "not_needed":
        lines.append("主链成功，备用训练无需启动。")
    else:
        lines.append("备用路径配置为使用原 Wan/Cosmos，从最近完整断点额外训练 20 步。")
    lines.append(f"角色事件日志：{output / (role + '_events.jsonl')}")
    return {
        "role": role,
        "outcome": status,
        "subject": f"[Spatial 看门狗] {label}",
        "body": "\n".join(lines),
    }


class OutcomeMailer:
    """One independent sender, durable outbox, deterministic message IDs, retry on failure."""

    def __init__(self, plan: dict, smtp_config: Path):
        self.plan = plan
        self.smtp_config = smtp_config
        self.root = Path(plan["output"]) / "outcome_email"
        self.events: dict[str, dict] = {}
        self.deliveries: dict[str, dict] = {}
        for key in ("armed", "primary-outcome", "fallback-outcome"):
            event = read_json(self.root / "outbox" / f"{key}.json")
            if event:
                self.events[key] = event
                self.deliveries[key] = read_json(self.root / "delivery" / f"{key}.json")

    def persist(self, path: Path, value: dict) -> None:
        try:
            atomic_json(path, value)
        except OSError as exc:
            # Retain the event/delivery in memory and keep sending if disk is full.
            watchdog_warning(
                str(path),
                f"Outcome email state unavailable: {path}; {mail_error_details(exc)}",
            )

    def enqueue(self, key: str, event: dict) -> None:
        if key in self.events:
            return
        digest = hashlib.sha256(f"{self.plan['output']}:{key}".encode()).hexdigest()
        event = {
            **event,
            "event_id": key,
            "created_utc": utc(),
            "message_id": f"<{digest}@spatial-watchdog.local>",
        }
        self.events[key] = event
        self.deliveries[key] = {"status": "queued", "attempts": 0}
        self.persist(self.root / "outbox" / f"{key}.json", event)
        self.persist(self.root / "delivery" / f"{key}.json", self.deliveries[key])
        watchdog_print(
            f"Outcome email queued: {key}; outcome={event.get('outcome', 'armed')}"
        )

    def confirm_armed(self) -> None:
        self.enqueue(
            "armed",
            {
                "subject": "[Spatial 看门狗] 两条链的结果邮件已启用（测试）",
                "body": (
                    "这是结果通知的真实 SMTP 测试，不代表训练已经结束。\n"
                    "主链和备用链成功、失败、意外退出、撤销均会通知；主链成功后备用无需启动也会通知。\n"
                    "邮件发送失败会在独立进程内重试，不改变训练或接续调度。\n"
                    f"节点：{socket.gethostname()}\n计划：{self.plan['output']}\n"
                    f"UTC：{utc()}\n"
                ),
            },
        )

    def scan(self) -> None:
        for role in ROLES:
            if role + "-outcome" in self.events:
                continue
            try:
                result = observe_outcome(self.plan, role)
                if result is not None:
                    self.enqueue(
                        role + "-outcome", outcome_event(self.plan, role, result)
                    )
            except (OSError, ValueError, KeyError) as exc:
                watchdog_warning(
                    role,
                    f"Outcome observation will retry: role={role}; {mail_error_details(exc)}",
                )

    def send_due(self, now: float | None = None) -> None:
        now = time.time() if now is None else now
        for key, event in self.events.items():
            delivery = self.deliveries[key]
            if (
                delivery.get("status") == "sent"
                or delivery.get("next_attempt_at", 0) > now
            ):
                continue
            attempts = delivery.get("attempts", 0) + 1
            delay = min(300, 30 * 2 ** min(attempts - 1, 4))
            delivery = {
                "status": "sending",
                "attempts": attempts,
                "last_attempt_utc": utc(),
                "next_attempt_at": now + delay,
            }
            self.deliveries[key] = delivery
            path = self.root / "delivery" / f"{key}.json"
            self.persist(path, delivery)
            try:
                settings = load_mail_settings(self.smtp_config)
                send_mail(
                    settings, event["subject"], event["body"], event["message_id"]
                )
            except (OSError, ValueError, TypeError, KeyError, smtplib.SMTPException) as exc:
                # Never log SMTP replies, addresses, credentials or exception text.
                delivery.update(status="retrying", error=mail_error_details(exc))
                watchdog_print(
                    f"Outcome email retry scheduled: {key}; {delivery['error']}"
                )
            else:
                delivery.update(status="sent", relay_accepted_utc=utc())
                delivery.pop("next_attempt_at", None)
                watchdog_print(f"Outcome email accepted by SMTP relay: {key}")
            self.persist(path, delivery)

    def complete(self) -> bool:
        return all(role + "-outcome" in self.events for role in ROLES) and all(
            value.get("status") == "sent" for value in self.deliveries.values()
        )

    def status(self) -> dict:
        return {
            "status": "complete" if self.complete() else "watching",
            "pid": os.getpid(),
            "updated_utc": utc(),
            "outcomes": {
                key: event.get("outcome", "armed") for key, event in self.events.items()
            },
            "deliveries": self.deliveries,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--smtp-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--send-armed",
        action="store_true",
        help="Send one clearly labelled activation/test email",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Observe and attempt delivery once, retaining pending mail",
    )
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if Path("/proc/sys/kernel/random/boot_id").read_text().strip() != plan["boot_id"]:
        raise RuntimeError(
            "Run this notifier on the execution node recorded in the plan"
        )
    root = Path(plan["output"]) / "outcome_email"
    root.mkdir(parents=True, exist_ok=True)
    with (root / "notifier.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        mailer = OutcomeMailer(plan, args.smtp_config)
        if args.send_armed:
            mailer.confirm_armed()
        watchdog_print(
            f"Outcome email watcher armed: {plan['output']}; interval=2s; pid={os.getpid()}"
        )
        last_report = float("-inf")
        while True:
            mailer.scan()
            mailer.send_due()
            if time.monotonic() - last_report >= 15 or mailer.complete() or args.once:
                mailer.persist(root / "state.json", mailer.status())
                last_report = time.monotonic()
            if mailer.complete() or args.once:
                return 0
            time.sleep(2)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        # Config/auth errors are handled and sanitized inside send_due.
        raise SystemExit(f"Outcome notifier stopped: {type(exc).__name__}") from None
