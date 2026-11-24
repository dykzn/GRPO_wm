"""Reusable per-rank CUDA cache for the unchanged Wan and Cosmos trainers.

Each training rank owns exactly one GPU. Reservations live in that rank's
native allocator and are available to its model, gradients and optimizer.
Operational settings are audited separately from checkpoint training identity.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import logging
import math
import os
import subprocess
import time
import xml.etree.ElementTree as ET
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

GIB = 1024**3
MIB = 1024**2
ALIGNMENT = 2 * MIB
LOG = logging.getLogger(__name__)


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    if value.lower() not in {"0", "1", "false", "true", "no", "yes"}:
        raise ValueError(f"{name} must be a boolean")
    return value.lower() in {"1", "true", "yes"}


@dataclass(frozen=True)
class MemorySettings:
    enabled: bool = True
    headroom_gib: float = 3.5
    target_gib: float | None = None
    chunk_mib: int = 256
    log_every: int = 10

    def __post_init__(self):
        if not math.isfinite(self.headroom_gib) or self.headroom_gib <= 0:
            raise ValueError("memory headroom must be finite and positive")
        if self.target_gib is not None and (
            not math.isfinite(self.target_gib) or self.target_gib <= 0
        ):
            raise ValueError("memory target must be finite and positive")
        if self.chunk_mib < 2 or self.log_every < 1:
            raise ValueError("memory chunk must be >= 2 MiB; log interval must be positive")

    @classmethod
    def from_args(cls, args):
        enabled = getattr(args, "reserve_memory", None)
        if enabled is None:
            enabled = env_bool("WM_UPDATE_RESERVE_MEMORY", True)
        return cls(
            enabled=enabled,
            headroom_gib=float(getattr(args, "memory_headroom_gib", os.environ.get("WM_UPDATE_MEMORY_HEADROOM_GIB", "3.5"))),
            target_gib=getattr(args, "memory_target_gib", None),
            log_every=int(getattr(args, "memory_log_every", 10)),
        )

    def command_args(self) -> list[str]:
        result = [
            "--reserve-memory" if self.enabled else "--no-reserve-memory",
            "--memory-headroom-gib", str(self.headroom_gib),
            "--memory-log-every", str(self.log_every),
        ]
        if self.target_gib is not None:
            result += ["--memory-target-gib", str(self.target_gib)]
        return result


def add_memory_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--reserve-memory", action=argparse.BooleanOptionalAction,
        default=env_bool("WM_UPDATE_RESERVE_MEMORY", True),
        help="Retain reusable CUDA memory in each training rank (default: enabled)",
    )
    parser.add_argument(
        "--memory-headroom-gib", type=float,
        default=float(os.environ.get("WM_UPDATE_MEMORY_HEADROOM_GIB", "3.5")),
        help="Device memory left outside the training allocator, in GiB",
    )
    parser.add_argument(
        "--memory-target-gib", type=float,
        help="Optional device-usage target; default: total VRAM minus headroom",
    )
    parser.add_argument("--memory-log-every", type=int, default=10)


def reservation_target(reserved: int, free: int, total: int, settings: MemorySettings) -> int:
    """Budget the whole device, including CUDA/NCCL memory outside PyTorch."""
    headroom = int(settings.headroom_gib * GIB)
    if headroom >= total:
        raise ValueError("memory headroom must be smaller than the GPU")
    device_target = (
        int(settings.target_gib * GIB) if settings.target_gib is not None else total - headroom
    )
    if device_target + headroom > total:
        raise ValueError("memory target plus headroom exceeds physical VRAM")
    outside_allocator = max(0, total - free - reserved)
    wanted = min(reserved + max(0, free - headroom), max(0, device_target - outside_allocator))
    return max(reserved, wanted // ALIGNMENT * ALIGNMENT)


def physical_device(local_rank: int) -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return str(local_rank)
    devices = [part.strip() for part in visible.split(",")]
    count = int(os.environ.get("LOCAL_WORLD_SIZE", len(devices)))
    if len(devices) != count or len(set(devices)) != count or not 0 <= local_rank < count:
        raise ValueError("Memory reservation requires one distinct visible GPU per local rank")
    return devices[local_rank]


def device_process_snapshot(device: str, pid: int) -> dict:
    result = subprocess.run(
        ["nvidia-smi", "-q", "-x"], text=True, capture_output=True,
        check=True, timeout=15,
    )
    xml = ET.fromstring(result.stdout)
    for index, gpu in enumerate(xml.findall("gpu")):
        uuid = gpu.findtext("uuid", "")
        if device not in {str(index), uuid, gpu.findtext("minor_number", "")}:
            continue
        processes = []
        for process in gpu.findall("processes/process_info"):
            process_pid = int(process.findtext("pid", "-1"))
            processes.append({"pid": process_pid, "type": process.findtext("type", "")})
        return {
            "physical_gpu": str(index), "gpu_uuid": uuid,
            "foreign_pids": [p["pid"] for p in processes if p["pid"] != pid],
            "processes": processes,
        }
    raise ValueError(f"Requested GPU was not found in nvidia-smi: {device}")


class TrainingMemoryReservation:
    def __init__(self, torch_module, output: Path, settings: MemorySettings, model: str):
        self.torch, self.cuda = torch_module, torch_module.cuda
        self.settings, self.model = settings, model
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.device = self.local_rank
        self.physical = physical_device(self.local_rank)
        self.root = Path(output) / "memory_reservation"
        self.root.mkdir(parents=True, exist_ok=True)
        self.log = self.root / f"rank_{self.rank}.jsonl"
        self.initialized = False
        self.in_reservation = False
        self.optimizer_updates = 0
        self.has_backward = False
        self.once = set()
        self.last_snapshot = float("-inf")
        self.snapshot = {}
        self.policy_id = f"{time.time_ns()}_rank_{self.rank}"
        self.native_reset = self.cuda.reset_peak_memory_stats

    def write(self, value: dict) -> None:
        try:
            with self.log.open("a") as stream:
                stream.write(json.dumps(value, allow_nan=False) + "\n")
        except OSError as error:
            LOG.warning("Memory reservation log unavailable: %s", type(error).__name__)

    def initialize(self) -> None:
        if self.initialized:
            if self.cuda.current_device() != self.device:
                raise RuntimeError("A training rank changed its reserved CUDA device")
            return
        if self.cuda.current_device() != self.device:
            raise ValueError("CUDA device differs from LOCAL_RANK")
        if self.cuda.get_allocator_backend() != "native":
            raise ValueError("Memory reservation requires the native CUDA allocator")
        options = dict(
            part.split(":", 1)
            for part in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "").split(",")
            if ":" in part
        )
        if options.get("expandable_segments") != "True":
            raise ValueError("Memory reservation requires expandable_segments:True")
        if float(options.get("garbage_collection_threshold", "0")) != 0:
            raise ValueError("Memory reservation requires garbage_collection_threshold:0")
        self.snapshot = device_process_snapshot(self.physical, os.getpid())
        self.last_snapshot = time.monotonic()
        if self.snapshot["foreign_pids"]:
            raise RuntimeError(
                f"GPU {self.physical} has other processes {self.snapshot['foreign_pids']}; "
                "this rank will not start training on an occupied GPU"
            )
        self.initialized = True
        policy = {
            "schema": "wm-training-memory-v1", "model": self.model,
            "policy_id": self.policy_id,
            "settings": asdict(self.settings), "pid": os.getpid(),
            "rank": self.rank, "local_rank": self.local_rank,
            "visible_gpus": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "checkpoint_training_identity_unchanged": True,
            **self.snapshot,
        }
        path = self.root / f"policy_rank_{self.rank}.json"
        path.write_text(json.dumps(policy, indent=2) + "\n")
        (self.root / f"policy_{self.policy_id}.json").write_text(json.dumps(policy, indent=2) + "\n")
        self.retain("initialized")
        # Reservation buffers are not model tensors. Exclude their transient
        # allocated peak before the native trainer starts measuring its work.
        self.native_reset(self.device)
        LOG.warning(
            "[wm-memory-pool model=%s rank=%s] GPU=%s reserved=%.2f GiB; "
            "headroom=%.2f GiB; log=%s",
            self.model, self.rank, self.physical,
            self.cuda.memory_reserved(self.device) / GIB,
            self.settings.headroom_gib, self.log,
        )

    def record(self, phase: str, **extra) -> None:
        if not self.initialized:
            return
        now = time.monotonic()
        if now - self.last_snapshot >= 30:
            self.last_snapshot = now
            try:
                self.snapshot = device_process_snapshot(self.physical, os.getpid())
            except (OSError, ValueError, ET.ParseError, subprocess.SubprocessError) as error:
                self.snapshot = {"telemetry_error": type(error).__name__}
        free, total = self.cuda.mem_get_info(self.device)
        self.write({
            "time": time.time(), "phase": phase, "pid": os.getpid(),
            "rank": self.rank, "model": self.model,
            "policy_id": self.policy_id,
            "optimizer_updates_since_launch": self.optimizer_updates,
            "allocated_gib": self.cuda.memory_allocated(self.device) / GIB,
            "reserved_gib": self.cuda.memory_reserved(self.device) / GIB,
            "peak_allocated_gib": self.cuda.max_memory_allocated(self.device) / GIB,
            "peak_reserved_gib": self.cuda.max_memory_reserved(self.device) / GIB,
            "device_used_gib": (total - free) / GIB,
            "device_total_gib": total / GIB,
            "device_free_gib": free / GIB,
            "process_snapshot_age_seconds": max(0, now - self.last_snapshot),
            "headroom_gib": self.settings.headroom_gib,
            **self.snapshot, **extra,
        })

    def record_once(self, phase: str) -> None:
        if phase not in self.once:
            self.once.add(phase)
            self.record(phase)

    def retain(self, phase: str) -> None:
        if not self.initialized or self.in_reservation:
            return
        self.in_reservation = True
        buffers = []
        try:
            with self.cuda.device(self.device), self.cuda.stream(self.cuda.default_stream(self.device)):
                reserved = self.cuda.memory_reserved(self.device)
                free, total = self.cuda.mem_get_info(self.device)
                target = reservation_target(reserved, free, total, self.settings)
                error = None
                try:
                    while self.cuda.memory_reserved(self.device) < target:
                        gap = target - self.cuda.memory_reserved(self.device)
                        buffers.append(self.torch.empty(
                            min(self.settings.chunk_mib * MIB, gap),
                            dtype=self.torch.uint8, device=self.device,
                        ))
                except self.torch.OutOfMemoryError:
                    error = "OutOfMemoryError during optional cache reservation"
                    LOG.warning("%s rank %s: %s", self.model, self.rank, error)
                finally:
                    transient_peak = self.cuda.max_memory_allocated(self.device)
                    buffers.clear()
                if phase == "initialized" or self.cuda.memory_reserved(self.device) != reserved or error:
                    self.record(
                        phase, target_reserved_gib=target / GIB,
                        reservation_transient_peak_allocated_gib=transient_peak / GIB,
                        reservation_error=error,
                    )
        finally:
            buffers.clear()
            self.in_reservation = False

    def reset_step_peak(self, device=None) -> None:
        # Native trainers reset once per optimizer-update window. Top up BEFORE
        # that reset so temporary reservation buffers cannot inflate their loss
        # or real-tensor peak statistics. No fill runs between reset and metrics.
        if self.initialized:
            self.retain("before_step")
        self.native_reset(device)

    def after_optimizer(self) -> None:
        # DCP may initialize Adam moments with a synthetic optimizer.step().
        # Count only steps preceded by a real training backward.
        if not self.has_backward:
            return
        self.has_backward = False
        self.optimizer_updates += 1
        if self.optimizer_updates == 1 or self.optimizer_updates % self.settings.log_every == 0:
            self.record("after_optimizer")


@contextmanager
def memory_hooks(torch_module, pool):
    """Scope operational hooks to this worker; restore all APIs on exit."""
    from torch.optim.optimizer import register_optimizer_step_post_hook
    import torch.distributed.checkpoint as dcp
    import safetensors.torch as safetensors_torch

    cuda = torch_module.cuda
    with ExitStack() as stack:
        def patch(owner, name, replacement):
            previous = getattr(owner, name)
            setattr(owner, name, replacement)
            stack.callback(setattr, owner, name, previous)
            return previous

        native_set_device = cuda.set_device

        @functools.wraps(native_set_device)
        def set_device(device):
            result = native_set_device(device)
            pool.initialize()
            return result

        patch(cuda, "set_device", set_device)
        patch(cuda, "reset_peak_memory_stats", pool.reset_step_peak)
        patch(cuda.memory, "reset_peak_memory_stats", pool.reset_step_peak)

        def keep_cache():
            pool.record_once("explicit_empty_cache_suppressed")

        patch(cuda, "empty_cache", keep_cache)
        patch(cuda.memory, "empty_cache", keep_cache)
        native_backward = torch_module.Tensor.backward

        @functools.wraps(native_backward)
        def backward(tensor, *args, **kwargs):
            pool.record_once("after_first_forward")
            try:
                result = native_backward(tensor, *args, **kwargs)
            except BaseException:
                pool.record("backward_failed")
                raise
            pool.has_backward = True
            pool.record_once("after_first_backward")
            return result

        patch(torch_module.Tensor, "backward", backward)
        handle = register_optimizer_step_post_hook(lambda optimizer, args, kwargs: pool.after_optimizer())
        stack.callback(handle.remove)

        def io_hook(operation, function):
            @functools.wraps(function)
            def wrapped(*args, **kwargs):
                pool.record("before_" + operation)
                result = function(*args, **kwargs)
                pool.record("after_" + operation)
                return result
            return wrapped

        for name in ("save", "load"):
            patch(dcp, name, io_hook("checkpoint_" + name, getattr(dcp, name)))
        patch(safetensors_torch, "save_file", io_hook("runtime_export", safetensors_torch.save_file))
        try:
            yield pool
        except BaseException:
            try:
                pool.record("training_failed")
            except Exception as error:
                LOG.warning("Memory telemetry failed while handling training error: %s", type(error).__name__)
            raise
        else:
            pool.record("training_complete")


def train_with_memory(args, model: str, train_function):
    settings = MemorySettings.from_args(args)
    if not settings.enabled:
        return train_function(args)
    import torch

    pool = TrainingMemoryReservation(torch, args.output, settings, model)
    with memory_hooks(torch, pool):
        return train_function(args)
