"""Measure native actor forward/backward and optimizer CUDA high-water marks."""

from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path

GIB = 1024**3
logger = logging.getLogger(__name__)


def atomic_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class ActorUpdatePeak:
    """Reset after cache warmup; measure before post-update cache refill."""

    def __init__(self, actor, torch_module):
        self.actor = actor
        self.cuda = torch_module.cuda
        self.device = actor.device
        self.started = time.monotonic()
        self.start_optimizer_steps = actor.optimizer_steps
        self.expected = actor.cfg.actor.gpu_memory_profile.get(
            "expected_optimizer_updates"
        )
        self.path = (
            Path(actor.cfg.runner.logger.log_path)
            / "gpu_peak"
            / f"actor_rank_{actor._rank}.json"
        )
        self.micro_batches = []

    def counters(self) -> dict:
        free, total = self.cuda.mem_get_info(self.device)
        return {
            "allocated_bytes": self.cuda.memory_allocated(self.device),
            "reserved_bytes": self.cuda.memory_reserved(self.device),
            "peak_allocated_bytes": self.cuda.max_memory_allocated(self.device),
            "peak_reserved_bytes": self.cuda.max_memory_reserved(self.device),
            "device_free_bytes": free,
            "device_total_bytes": total,
        }

    def start(self) -> None:
        self.cuda.synchronize(self.device)
        self.cuda.reset_peak_memory_stats(self.device)
        self.initial = self.counters()

    def after_backward(self) -> None:
        self.cuda.synchronize(self.device)
        counters = self.counters()
        self.micro_batches.append(dict(index=len(self.micro_batches) + 1, **counters))
        logger.warning(
            "[backward-peak rank=%s micro=%s/%s] allocated_peak=%.2f GiB reserved_peak=%.2f GiB",
            self.actor._rank,
            len(self.micro_batches),
            self.actor.gradient_accumulation,
            counters["peak_allocated_bytes"] / GIB,
            counters["peak_reserved_bytes"] / GIB,
        )

    def finish(self, metrics=None, error=None) -> dict:
        counter_error = None
        try:
            self.cuda.synchronize(self.device)
            counters = self.counters()
        except (RuntimeError, ValueError) as exc:
            counters = None
            counter_error = f"{type(exc).__name__}: {exc}"
        steps = self.actor.optimizer_steps - self.start_optimizer_steps
        correct_count = self.expected is None or steps == int(self.expected)
        expected_backwards = (
            int(self.expected) * int(self.actor.gradient_accumulation)
            if self.expected is not None
            else None
        )
        correct_backwards = (
            expected_backwards is None or len(self.micro_batches) == expected_backwards
        )
        completed = (
            error is None
            and counters is not None
            and correct_count
            and correct_backwards
        )
        grad_norm = float((metrics or {}).get("actor/grad_norm", float("nan")))
        completed = completed and math.isfinite(grad_norm)
        report = {
            "status": "complete" if completed else "failed",
            "rank": int(self.actor._rank),
            "pid": os.getpid(),
            "device": str(self.device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "grpo_step": int(getattr(self.actor, "version", 0)),
            "micro_batch_size": int(self.actor.cfg.actor.micro_batch_size),
            "gradient_accumulation": int(self.actor.gradient_accumulation),
            "backward_calls_completed": len(self.micro_batches),
            "backward_calls_expected": expected_backwards,
            "optimizer_step_counter_delta": int(steps),
            "expected_optimizer_updates": self.expected,
            "grad_norm": grad_norm if math.isfinite(grad_norm) else None,
            "nonzero_finite_grad_norm": math.isfinite(grad_norm) and grad_norm > 0,
            "elapsed_seconds": time.monotonic() - self.started,
            "baseline": self.initial,
            "micro_batches": self.micro_batches,
            "update": counters,
            "backward_peak_allocated_gib": (
                self.micro_batches[-1]["peak_allocated_bytes"] / GIB
                if self.micro_batches
                else None
            ),
            "update_peak_allocated_gib": counters["peak_allocated_bytes"] / GIB
            if counters
            else None,
            "update_peak_reserved_gib": counters["peak_reserved_bytes"] / GIB
            if counters
            else None,
            "error": f"{type(error).__name__}: {error}" if error else counter_error,
            "interpretation": [
                "Allocator peaks cover this actor process, not Cosmos/rollout/other jobs.",
                "Reset follows before_training cache warmup; measurement ends before after_training refill.",
                "Reserved peak includes the existing CUDA allocator cache; allocated peak tracks tensors.",
                "Backward calls are the real native policy loss backward, even when advantages are zero.",
                "Optimizer counter counts attempts; inspect grad_norm for nonfinite or zero-gradient updates.",
            ],
        }
        atomic_report(self.path, report)
        logger.warning(
            "[update-peak rank=%s status=%s] allocated_peak=%s GiB reserved_peak=%s GiB; %s",
            self.actor._rank,
            report["status"],
            report["update_peak_allocated_gib"],
            report["update_peak_reserved_gib"],
            self.path,
        )
        if error is None and not completed:
            raise RuntimeError(
                f"GPU peak probe did not complete the expected update: {self.path}"
            )
        return report


def write_peak_summary(output: Path, world_size: int = 8) -> dict:
    root = output / "gpu_peak"
    actors = []
    missing = []
    for rank in range(world_size):
        path = root / f"actor_rank_{rank}.json"
        if path.is_file():
            actors.append(json.loads(path.read_text()))
        else:
            missing.append(rank)
    device_path = root / "device" / "summary.json"
    devices = json.loads(device_path.read_text()) if device_path.exists() else None
    complete = not missing and all(row["status"] == "complete" for row in actors)

    def maximum(key):
        values = [row[key] for row in actors if row.get(key) is not None]
        return max(values) if values else None

    report = {
        "status": "complete" if complete else "incomplete_or_failed",
        "actor_reports": actors,
        "missing_actor_ranks": missing,
        "max_actor_backward_allocated_gib": maximum("backward_peak_allocated_gib"),
        "max_actor_update_allocated_gib": maximum("update_peak_allocated_gib"),
        "max_actor_update_reserved_gib": maximum("update_peak_reserved_gib"),
        "device_monitor": devices,
        "max_observed_device_used_gib": (
            max(row["peak_used_mib"] for row in devices["gpus"]) / 1024
            if devices and devices.get("gpus")
            else None
        ),
    }
    atomic_report(root / "peak_summary.json", report)
    print(
        f"Backward/update memory test: {report['status']}. "
        f"Backward allocated peak={report['max_actor_backward_allocated_gib']} GiB; "
        f"update allocated peak={report['max_actor_update_allocated_gib']} GiB; "
        f"reserved peak={report['max_actor_update_reserved_gib']} GiB. "
        f"Report: {root / 'peak_summary.json'}",
        flush=True,
    )
    return report
