"""Cooperative memory admission and recovery for one colocated RLinf job.

This does not reserve a GPU against other users. It waits at safe boundaries,
retries pure inference calls, and publishes only completed recovery checkpoints.
No process outside the current run is signalled or modified.
"""

import csv
import json
import logging
import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

GIB = 1024**3
LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class GuardSettings:
    enabled: bool = False
    poll_seconds: float = 5.0
    stable_checks: int = 2
    foreign_limit_mib: float = 256.0
    imagination_free_gib: float = 38.0
    update_free_gib: float = 12.0
    checkpoint_each_step: bool = True
    retain_imagination_cache: bool = True

    @classmethod
    def from_config(cls, config):
        values = config or {}
        obj = cls(
            **{
                name: values[name]
                for name in cls.__dataclass_fields__
                if name in values
            }
        )
        if obj.stable_checks < 1 or int(obj.stable_checks) != obj.stable_checks:
            raise ValueError("memory_guard.stable_checks must be a positive integer")
        for name in (
            "poll_seconds",
            "foreign_limit_mib",
            "imagination_free_gib",
            "update_free_gib",
        ):
            value = getattr(obj, name)
            if (
                not math.isfinite(value)
                or value < 0
                or (name == "poll_seconds" and value == 0)
            ):
                raise ValueError(f"Invalid memory_guard.{name}")
        return obj


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def process_belongs_to_run(pid, run):
    """Inspect only the run tag; never log or copy other environment values."""
    try:
        entries = (Path("/proc") / str(pid) / "environ").read_bytes().split(b"\0")
    except (OSError, ValueError):
        return False
    expected = os.fsencode("SPATIAL_WOVR_OUTPUT=" + str(run))
    return expected in entries


def visible_devices(local_index=None):
    devices = [
        x.strip() for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if x.strip()
    ]
    if local_index is not None:
        devices = [devices[local_index]]
    return devices


def query_snapshot(
    devices, run, *, command=subprocess.run, owns=process_belongs_to_run
):
    def query(fields, kind):
        result = command(
            ["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        return list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))

    gpu_rows = query("index,uuid,memory.free", "gpu")
    wanted = set(map(str, devices))
    selected = {
        uuid: {
            "index": int(index),
            "free_mib": float(free),
            "foreign_mib": 0.0,
            "foreign": [],
        }
        for index, uuid, free in gpu_rows
        if index in wanted or uuid in wanted
    }
    if len(selected) != len(wanted):
        raise ValueError("GPU memory query did not include every requested device")
    for row in query("gpu_uuid,pid,used_memory", "compute-apps"):
        if not row or row[0] not in selected:
            continue
        uuid, pid, used = row
        pid, used = int(pid), float(used)
        if not owns(pid, run):
            selected[uuid]["foreign_mib"] += used
            selected[uuid]["foreign"].append({"pid": pid, "used_mib": used})
    return list(selected.values())


class MemoryGuard:
    def __init__(
        self, settings, devices, run, *, probe=query_snapshot, sleep=time.sleep
    ):
        if not run:
            raise ValueError("Memory protection requires SPATIAL_WOVR_OUTPUT")
        self.settings = settings
        self.devices = list(map(str, devices))
        self.run = str(run)
        self.probe = probe
        self.sleep = sleep
        self.last_clear = 0.0

    def wait(self, phase, *, required_free_gib=0.0, force=False, require_stable=False):
        if not self.settings.enabled:
            return False
        if (
            not force
            and not required_free_gib
            and time.monotonic() - self.last_clear < 5
        ):
            return False
        stable = 0
        blocked = require_stable
        last_report = 0.0
        while True:
            error = None
            try:
                rows = self.probe(self.devices, self.run)
                clear = all(
                    row["foreign_mib"] <= self.settings.foreign_limit_mib
                    and row["free_mib"] >= required_free_gib * 1024
                    for row in rows
                )
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                rows, clear, error = [], False, type(exc).__name__
            stable = stable + 1 if clear else 0
            if stable >= (self.settings.stable_checks if blocked else 1):
                self.last_clear = time.monotonic()
                if blocked:
                    self._report(phase, "ready", rows, None)
                return blocked
            blocked = True
            if time.monotonic() - last_report >= 30 or last_report == 0:
                self._report(phase, "waiting", rows, error)
                last_report = time.monotonic()
            self.sleep(self.settings.poll_seconds)

    def _report(self, phase, status, rows, error):
        value = {
            "status": status,
            "phase": phase,
            "pid": os.getpid(),
            "time": time.time(),
            "devices": rows,
            "query_error": error,
        }
        LOG.warning(
            "[memory-guard] %s phase=%s devices=%s query_error=%s",
            status,
            phase,
            rows,
            error,
        )
        try:
            atomic_json(
                Path(self.run) / "memory_guard" / f"worker_{os.getpid()}.json", value
            )
        except OSError as exc:
            LOG.warning("Cannot persist memory guard state: %s", type(exc).__name__)

    def inference(self, function, *, phase, release_cache=lambda: None):
        """Retry a fixed-input call before any environment history is committed."""
        uncontended_failures = 0
        self.wait(phase)
        while True:
            message = None
            try:
                return function()
            except Exception as exc:
                if not is_cuda_oom(exc):
                    raise
                message = str(exc)
                LOG.warning(
                    "[memory-guard] inference OOM phase=%s; checking contention", phase
                )
            # Outside the except block: failed inference frames no longer retain
            # intermediate tensors through the exception traceback.
            release_cache()
            waited = self.wait(phase, force=True)
            uncontended_failures = 0 if waited else uncontended_failures + 1
            if uncontended_failures > 1:
                raise RuntimeError(
                    "Repeated CUDA out of memory without observable contention: "
                    + message
                )
            self.sleep(self.settings.poll_seconds)


def is_cuda_oom(error):
    return (
        "cuda out of memory" in str(error).lower()
        or "cuda error: out of memory" in str(error).lower()
    )


def guard_for_env(env):
    settings = GuardSettings.from_config(env.cfg.get("memory_guard", {}))
    if not settings.enabled or getattr(env.device, "type", str(env.device)) != "cuda":
        return None
    guard = getattr(env, "_contention_guard", None)
    if guard is None:
        import torch

        index = env.device.index
        if index is None:
            index = torch.cuda.current_device()
        guard = MemoryGuard(
            settings, visible_devices(index), os.environ.get("SPATIAL_WOVR_OUTPUT")
        )
        env._contention_guard = guard
    return guard


def guarded_inference(env, function, phase):
    guard = guard_for_env(env)
    if guard is None:
        return function()
    import torch

    return guard.inference(function, phase=phase, release_cache=torch.cuda.empty_cache)


def retain_imagination_cache(env):
    guard = guard_for_env(env)
    if guard is None or not guard.settings.retain_imagination_cache:
        return
    pool = getattr(env, "_imagination_cache_reservation", None)
    if pool is None:
        import torch
        from spatial_grpo_memory import CudaCacheReservation

        pool = CudaCacheReservation(torch, env.device, 0.95, 4.0, "wan")
        env._imagination_cache_reservation = pool
    # Called only after both views have run, so the Cosmos allocator has
    # already established its own high-water mark on the first chunk.
    pool.retain("imagination")


def configure_recovery(cfg):
    settings = GuardSettings.from_config(cfg.runner.get("memory_guard", {}))
    if settings.enabled and settings.checkpoint_each_step:
        steps = set(cfg.actor.fsdp_config.get("save_training_state_at_steps", []))
        steps.update(range(1, int(cfg.runner.max_steps) + 1))
        cfg.actor.fsdp_config.save_training_state_at_steps = sorted(steps)


def checkpoint_path(cfg, step):
    return (
        Path(cfg.runner.logger.log_path)
        / cfg.runner.logger.experiment_name
        / "checkpoints"
        / f"global_step_{step}"
    )


def checkpoint_complete(path, world_size):
    path = Path(path)
    required = [
        path / "actor/model_state_dict/full_weights.pt",
        path / "actor/dcp_checkpoint/.metadata",
    ]
    required += [
        path / f"actor/dcp_checkpoint/__{rank}_0.distcp" for rank in range(world_size)
    ]
    return all(file.is_file() and file.stat().st_size > 0 for file in required)


class MemoryGuardRunnerMixin:
    def _memory_guard(self):
        settings = GuardSettings.from_config(self.cfg.runner.get("memory_guard", {}))
        if not settings.enabled:
            return None
        if not hasattr(self, "_contention_guard"):
            self._contention_guard = MemoryGuard(
                settings, visible_devices(), os.environ.get("SPATIAL_WOVR_OUTPUT")
            )
        return self._contention_guard

    def update_rollout_weights(self):
        result = super().update_rollout_weights()
        guard = self._memory_guard()
        if guard is not None:
            guard.wait(
                "imagination_admission",
                required_free_gib=guard.settings.imagination_free_gib,
                force=True,
            )
        return result

    def _wait_for_update_memory(self):
        guard = self._memory_guard()
        if guard is not None:
            guard.wait(
                "update_admission",
                required_free_gib=guard.settings.update_free_gib,
                force=True,
            )

    def _maybe_eval_and_checkpoint(self, step):
        metrics = super()._maybe_eval_and_checkpoint(step)
        guard = self._memory_guard()
        if guard is not None and guard.settings.checkpoint_each_step:
            path = checkpoint_path(self.cfg, self.global_step)
            if not checkpoint_complete(path, len(guard.devices)):
                self._save_checkpoint()
            if not checkpoint_complete(path, len(guard.devices)):
                raise RuntimeError(f"Incomplete recovery checkpoint: {path}")
            # Written only after all rank saves AND any scheduled evaluation
            # have returned. A failed/partial save is never a resume candidate.
            atomic_json(
                path / "recovery_complete.json",
                {
                    "step": self.global_step,
                    "world_size": len(guard.devices),
                    "time": time.time(),
                },
            )
        return metrics
