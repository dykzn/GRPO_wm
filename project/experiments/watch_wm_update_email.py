"""Notify success/failure of one formal collection + Wan/Cosmos update queue."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

from queue_spatial_wm_update import process_alive
from spatial_followup_email import OutcomeMailer, read_json
from watch_training_email import (
    DEFAULT_CONFIG,
    mail_error_details,
    watchdog_print,
    watchdog_warning,
)


def utc():
    return datetime.now(UTC).isoformat()


def terminal_result(plan: dict) -> dict | None:
    output = Path(plan["output"])

    def recorded():
        state = read_json(output / "state.json")
        if state.get("status") == "failed":
            return {**state, "outcome": "failed"}
        if state.get("status") == "complete":
            collection = read_json(output / "collection/collection.json")
            pair = read_json(output / f"pair_{plan['steps']}/pair.json")
            if (
                collection.get("summary", {}).get("episodes") != 10 * plan["per_task"]
                or pair.get("wan_step") != plan["steps"]
                or pair.get("cosmos_step") != plan["steps"]
            ):
                return {
                    **state,
                    "outcome": "failed",
                    "error": "Queue completed but final collection/model-pair checks failed",
                }
            return {**state, "outcome": "complete"}
        return None

    result = recorded()
    if result is not None:
        return result
    identity = read_json(output / "worker_identity.json")
    if not identity or process_alive(identity):
        return None
    # A normal exit can occur between the first state read and the PID check.
    result = recorded()
    if result is not None:
        return result
    return {
        **read_json(output / "state.json"),
        "outcome": "failed",
        "error": "Queue supervisor exited without a terminal result (PID/start time no longer matches)",
    }


def notification(plan: dict, result: dict) -> dict:
    output = Path(plan["output"])
    label = "成功" if result["outcome"] == "complete" else "失败"
    stage = result.get("stage", result.get("status", "unknown"))
    events = output / "events.jsonl"
    if events.exists():
        for line in reversed(events.read_text().splitlines()):
            try:
                item = json.loads(line)
            except ValueError:
                continue
            if "stage" in item:
                stage = item["stage"]
                break
    lines = [
        f"LIBERO Spatial 正式 world model 更新：{label}",
        f"节点：{plan['expected_node_ip']}",
        f"UTC：{utc()}",
        f"策略：GRPO 第 {plan['target_step']} 步；{plan['policy_checkpoint']}",
        f"采集目标：10 个任务 × {plan['per_task']} 条 = {10 * plan['per_task']} 条",
        f"模型更新：Wan 四卡 + Cosmos 四卡；各 {plan['steps']} 步；每 {plan['save_interval']} 步保存",
        f"最后阶段：{stage}",
        f"输出：{output}",
        f"日志：{output / 'logs/queue.log'}",
    ]
    collection = read_json(output / "collection/collection.json")
    if collection:
        lines.append(
            f"采集统计：{json.dumps(collection.get('summary', {}), ensure_ascii=False)}"
        )
    if result.get("error"):
        lines.append(f"错误：{str(result['error'])[:2000]}")
    for model in ("wan", "cosmos"):
        path = output / "train" / model / "metrics.jsonl"
        if path.exists():
            rows = path.read_text().splitlines()
            if rows:
                try:
                    last = json.loads(rows[-1])
                except ValueError:
                    continue
                lines.append(
                    f"{model} 最后记录：{json.dumps(last, ensure_ascii=False)}"
                )
    if result["outcome"] == "complete":
        lines.append(f"模型对：{output / ('pair_' + str(plan['steps'])) / 'pair.json'}")
    return {
        "role": "world_model_update",
        "outcome": result["outcome"],
        "subject": f"[Spatial WM] 500 条采集与模型更新{label}",
        "body": "\n".join(lines),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--smtp-config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if (
        Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        != plan["execution_boot_id"]
    ):
        parser.error("Run this watcher on the formal queue's recorded GPU node")
    root = Path(plan["output"]) / "outcome_email"
    root.mkdir(exist_ok=True)
    with (root / "notifier.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        mailer = OutcomeMailer(plan, args.smtp_config)
        # The primary outbox slot represents this single main WM queue.
        key = "primary-outcome"
        watchdog_print(
            f"WM UPDATE WATCHDOG ARMED: {plan['output']}; pid={os.getpid()}; interval=2s"
        )
        reported = float("-inf")
        while True:
            if key not in mailer.events:
                try:
                    result = terminal_result(plan)
                    if result is not None:
                        mailer.enqueue(key, notification(plan, result))
                except (OSError, ValueError, KeyError) as exc:
                    watchdog_warning(
                        "observe",
                        f"WM outcome observation will retry: {mail_error_details(exc)}",
                    )
            mailer.send_due()
            sent = mailer.deliveries.get(key, {}).get("status") == "sent"
            if time.monotonic() - reported >= 15 or sent:
                mailer.persist(
                    root / "state.json",
                    {
                        "status": "notified" if sent else "watching",
                        "pid": os.getpid(),
                        "updated_utc": utc(),
                        "delivery": mailer.deliveries.get(key),
                        "outcome": mailer.events.get(key, {}).get("outcome"),
                    },
                )
                reported = time.monotonic()
            if sent:
                return
            time.sleep(2)


if __name__ == "__main__":
    main()
