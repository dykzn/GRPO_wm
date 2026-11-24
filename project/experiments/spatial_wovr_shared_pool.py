"""Run Wan in its colocated actor, retaining one native CUDA allocator cache.

The environment RPC carries CPU arrays only. CUDA tensors, pointers and model
objects never cross processes. This module does not wait for or stop other jobs.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import secrets
import subprocess
import threading
import time
import xml.etree.ElementTree as ET
from contextlib import contextmanager, nullcontext
from multiprocessing.connection import Client, Listener
from multiprocessing.context import AuthenticationError
from pathlib import Path

import numpy as np
from PIL import Image

GIB = 1024**3
MIB = 1024**2
PROTOCOL = 1


def configure_shared_pool(cfg, run_dir):
    """Create private per-run RPC credentials before launching any GPU workers."""
    settings = cfg.actor.get("shared_wan_pool", {})
    if not settings.get("enabled", False):
        return
    if int(cfg.cluster.num_nodes) != 1 or int(cfg.rollout.pipeline_stage_num) != 1:
        raise ValueError("Shared Wan pool requires one node and one pipeline stage")
    if not cfg.actor.get("memory_pool", {}).get("enabled", False):
        raise ValueError("Shared Wan pool requires actor.memory_pool.enabled=true")
    if cfg.actor.get("enable_offload", False) or cfg.rollout.enable_offload:
        raise ValueError("Shared Wan pool requires resident actor and rollout")
    if not cfg.env.train.enable_offload or not cfg.env.train.get(
        "enable_init_offload", True
    ):
        raise ValueError(
            "Shared Wan pool requires environment offload, including initialization"
        )
    if cfg.runner.get("memory_guard", {}).get("enabled", False):
        raise ValueError(
            "Disable the separate-process memory guard for the shared Wan pool"
        )
    for key in ("target_job_gib", "device_headroom_gib", "rpc_timeout_seconds"):
        value = float(settings[key])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid shared_wan_pool.{key}")
    if settings.get("peer_allocator_gib") and cfg.runner.get("only_eval", False):
        # Stand-alone simulation evaluation has no Wan environment worker.
        settings.peer_allocator_gib = {}
    peers = settings.get("peer_allocator_gib", {})
    if peers:
        if set(peers) != {"cosmos", "rollout", "env"} or any(
            not math.isfinite(float(v)) or float(v) <= 0.25 for v in peers.values()
        ):
            raise ValueError("Invalid shared pool peer allocator budgets")
        growth = float(settings.get("driver_growth_gib", -1))
        if not math.isfinite(growth) or growth < 0:
            raise ValueError("Invalid driver growth allowance")
    startup = float(settings.get("startup_target_job_gib", 65))
    if not math.isfinite(startup) or startup <= 0:
        raise ValueError("Invalid shared_wan_pool.startup_target_job_gib")
    alloc = dict(
        item.split(":", 1)
        for item in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "").split(",")
        if ":" in item
    )
    if alloc.get("backend") != "native" or alloc.get("expandable_segments") != "True":
        raise ValueError(
            "Shared Wan pool requires native allocator with expandable_segments:True"
        )
    if float(alloc.get("garbage_collection_threshold", 0)) != 0:
        raise ValueError("Shared Wan pool requires garbage_collection_threshold:0")
    # Use the launcher's short TMPDIR; AF_UNIX paths must fit in 107 bytes.
    root = Path(os.environ["TMPDIR"]) / "wan-pool"
    root.mkdir(mode=0o700, parents=False, exist_ok=False)
    if len(os.fsencode(str(root / "rank-999.sock"))) >= 108:
        raise ValueError("Shared Wan socket path is too long; use a shorter TMPDIR")
    (root / "authkey").write_bytes(secrets.token_bytes(32))
    (root / "authkey").chmod(0o600)
    settings.rpc_root = str(root)
    settings.log_dir = str(Path(run_dir) / "shared_wan_pool")


def endpoint(settings, rank):
    root = Path(settings["rpc_root"])
    return str(root / f"rank-{int(rank)}.sock"), (root / "authkey").read_bytes()


def _cpu_wire(value):
    """Reject tensors and object arrays instead of invoking CUDA/pickle reducers."""
    if value is None or type(value) in (str, bytes, bool, int, float):
        return
    if isinstance(value, np.ndarray) and not value.dtype.hasobject:
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _cpu_wire(item)
        return
    if isinstance(value, dict) and all(type(k) is str for k in value):
        for item in value.values():
            _cpu_wire(item)
        return
    raise TypeError(
        f"Shared Wan RPC accepts only CPU arrays/scalars, got {type(value).__name__}"
    )


class SharedWanProxy:
    """An environment-side pipeline handle with no CUDA model or allocator."""

    def __init__(self, settings, rank):
        self.settings, self.rank = settings, int(rank)
        self.loaded = False
        self.warmed = False

    def request(self, operation, **payload):
        address, key = endpoint(self.settings, self.rank)
        message = {"protocol": PROTOCOL, "operation": operation, **payload}
        _cpu_wire(message)
        # Credentials and the socket live in a private 0700 per-run directory.
        # Connection.send uses normal pickle, never multiprocessing CUDA IPC.
        with Client(address, family="AF_UNIX", authkey=key) as connection:
            connection.send(message)
            if not connection.poll(float(self.settings["rpc_timeout_seconds"])):
                raise TimeoutError(
                    f"Shared Wan rank {self.rank} timed out during {operation}"
                )
            reply = connection.recv()
        _cpu_wire(reply)
        if not reply["ok"]:
            raise RuntimeError(
                f"Shared Wan rank {self.rank}, {operation}: {reply['error']}"
            )
        return reply["result"]

    def onload(self):
        if not self.loaded:
            self.request("onload")
            self.loaded = True

    def offload(self):
        # Env initialization runs before actor initialization. No RPC is needed
        # until this environment has acquired a GPU-resident Wan lease.
        if self.loaded:
            self.request("offload")
            self.loaded = False

    def generate(self, kwargs, rows, total):
        if not self.loaded:
            raise RuntimeError("Shared Wan generation requires onload first")
        values = dict(kwargs)
        values.pop("progress_bar_cmd", None)  # Process-local callback.
        action = values["action"]
        if not isinstance(action, np.ndarray):
            action = action.detach().cpu().numpy()
        values["action"] = np.ascontiguousarray(action)
        values["input_image"] = [
            np.asarray(im, dtype=np.uint8) for im in values["input_image"]
        ]
        values["input_image4"] = [
            [np.asarray(im, dtype=np.uint8) for im in images]
            for images in values["input_image4"]
        ]
        result = self.request(
            "generate", kwargs=values, rows=list(rows), total=int(total)
        )
        return [[Image.fromarray(frame) for frame in frames] for frames in result]

    def complete_first_chunk(self):
        if not self.warmed:
            self.request("chunk_complete")
            self.warmed = True


class SharedWanOwner:
    """Serialize Wan calls and actor updates around one process-wide allocator."""

    def __init__(
        self,
        pipeline,
        device,
        *,
        execution=nullcontext,
        retain=lambda phase: None,
        activate=lambda: None,
    ):
        self.pipeline, self.device = pipeline, device
        self.execution, self.retain = execution, retain
        self.activate = activate
        self.lock = threading.RLock()
        self.loaded = False

    @contextmanager
    def actor_phase(self, name):
        with self.lock:
            if self.loaded:
                raise RuntimeError(
                    f"Actor {name} attempted before Wan offload acknowledgement"
                )
            yield

    def dispatch(self, message):
        if message.get("protocol") != PROTOCOL:
            raise ValueError("Incompatible shared Wan RPC protocol")
        operation = message["operation"]
        with self.lock, self.execution():
            if operation == "chunk_complete":
                if not self.loaded:
                    raise RuntimeError(
                        "Chunk completion requires a GPU-resident Wan lease"
                    )
                self.activate()
                self.retain("first_dual_view_chunk_complete")
                return None
            if operation == "onload":
                if not self.loaded:
                    self.pipeline.dit = self.pipeline.dit.to(self.device)
                    self.pipeline.vae = self.pipeline.vae.to(self.device)
                    self.loaded = True
                self.retain("wan_onload")
                return None
            if operation == "offload":
                if self.loaded:
                    self.pipeline.vae = self.pipeline.vae.to("cpu")
                    self.pipeline.dit = self.pipeline.dit.to("cpu")
                    self.loaded = False
                self.retain("wan_offload")
                return None
            if operation != "generate" or not self.loaded:
                raise RuntimeError("Generation requires a GPU-resident Wan lease")
            import torch
            from rlinf.envs.world_model.generation_batch import original_slot_noise

            kwargs = dict(message["kwargs"])
            kwargs["action"] = torch.from_numpy(kwargs["action"]).to(self.device)
            kwargs["input_image"] = [
                Image.fromarray(im) for im in kwargs["input_image"]
            ]
            kwargs["input_image4"] = [
                [Image.fromarray(im) for im in frames]
                for frames in kwargs["input_image4"]
            ]
            kwargs["progress_bar_cmd"] = lambda values: values
            # The mask and seed assignment belong to the real pipeline owner.
            with (
                torch.no_grad(),
                original_slot_noise(self.pipeline, message["rows"], message["total"]),
            ):
                output = self.pipeline(**kwargs)
            return [
                [np.asarray(frame, dtype=np.uint8) for frame in frames]
                for frames in output
            ]


class SharedWanServer:
    """A daemon thread inside the actor; it never creates another GPU process."""

    def __init__(self, settings, rank, owner):
        self.settings, self.rank, self.owner = settings, rank, owner
        address, key = endpoint(settings, rank)
        self.listener = Listener(address, family="AF_UNIX", authkey=key)
        self.thread = threading.Thread(
            target=self._serve, name=f"wan-pool-{rank}", daemon=True
        )
        self.thread.start()

    def _serve(self):
        try:
            while True:
                try:
                    with self.listener.accept() as connection:
                        message = connection.recv()
                        try:
                            _cpu_wire(message)
                            if message == {
                                "protocol": PROTOCOL,
                                "operation": "shutdown",
                            }:
                                connection.send({"ok": True, "result": None})
                                return
                            result = self.owner.dispatch(message)
                            _cpu_wire(result)
                            reply = {"ok": True, "result": result}
                        except Exception as error:
                            logging.getLogger(__name__).exception(
                                "Shared Wan rank %s request failed", self.rank
                            )
                            # Never pickle exception objects with CUDA traceback state.
                            reply = {
                                "ok": False,
                                "error": f"{type(error).__name__}: {error}",
                            }
                        connection.send(reply)
                except (OSError, EOFError, AuthenticationError):
                    logging.getLogger(__name__).exception(
                        "Shared Wan rank %s RPC disconnected", self.rank
                    )
        finally:
            self.listener.close()

    def close(self):
        SharedWanProxy(self.settings, self.rank).request("shutdown")
        self.thread.join(timeout=5)


def job_device_memory(device, run):
    """Measure this task's resident memory, separately from unrelated processes."""
    from spatial_wovr_memory_guard import process_belongs_to_run, visible_devices

    def query(kind, fields):
        output = subprocess.run(
            ["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout
        return list(csv.reader(output.splitlines(), skipinitialspace=True))

    if isinstance(device, int):
        local_index = device
    elif isinstance(device, str):
        local_index = int(device.split(":", 1)[1]) if ":" in device else 0
    else:
        local_index = device.index or 0
    physical = visible_devices(local_index)[0]
    matches = [
        row
        for row in query("gpu", "index,uuid,memory.total,memory.used")
        if physical in row[:2]
    ]
    if len(matches) != 1:
        raise ValueError("Cannot map actor CUDA device to a physical GPU")
    _, uuid, total, used = matches[0]
    # Compute-only inventory omits LIBERO's graphics-only EGL contexts.
    xml = subprocess.run(
        ["nvidia-smi", "-q", "-x"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    ).stdout
    root = ET.fromstring(xml)
    gpu = next(
        (gpu for gpu in root.findall("gpu") if gpu.findtext("uuid") == uuid), None
    )
    if gpu is None:
        raise ValueError("NVIDIA XML did not include the actor's physical GPU")
    by_pid, graphics = {}, set()
    for process in gpu.findall("processes/process_info"):
        pid = int(process.findtext("pid"))
        memory = int(float(process.findtext("used_memory").split()[0]) * MIB)
        by_pid[pid] = max(by_pid.get(pid, 0), memory)
        if "G" in (process.findtext("type") or ""):
            graphics.add(pid)
    own, foreign, own_graphics = 0, 0, 0
    for pid, memory in by_pid.items():
        if process_belongs_to_run(pid, run):
            own += memory
            if pid in graphics:
                own_graphics += memory
        else:
            foreign += memory
    return {
        "job_used": own,
        "job_graphics_used": own_graphics,
        "foreign_used": foreign,
        "device_total": int(float(total) * MIB),
        "device_used": int(float(used) * MIB),
    }


def shared_reservation_target(reserved, free, job_used, target_job, headroom):
    """Top up the shared allocator to a whole-task target, preserving headroom."""
    return reserved + max(0, min(target_job - job_used, free - headroom)) // (
        2 * MIB
    ) * (2 * MIB)


def reserve_peer_allocator(torch, device, role, rank, budget_gib, log_dir):
    """Warm this worker's own CUDA allocator before the actor sizes its pool."""
    device = torch.device("cuda", device) if isinstance(device, int) else torch.device(device)
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    limit = int(float(budget_gib) * GIB)
    slack = 256 * MIB  # Allow small blocks and other streams to grow within the cap.
    if limit <= slack or torch.cuda.get_allocator_backend() != "native":
        raise ValueError(f"Invalid CUDA allocator budget for {role}: {budget_gib} GiB")
    with torch.cuda.device(device), torch.cuda.stream(torch.cuda.default_stream(device)):
        free, total = torch.cuda.mem_get_info(device)
        if limit > total or torch.cuda.memory_reserved(device) > limit:
            raise ValueError(f"{role} CUDA allocator already exceeds its configured limit")
        torch.cuda.set_per_process_memory_fraction(limit / total, device)
        held = []
        while torch.cuda.memory_reserved(device) < limit - slack:
            gap = limit - slack - torch.cuda.memory_reserved(device)
            held.append(torch.empty(min(gap, 256 * MIB), dtype=torch.uint8, device=device))
        held.clear()
        retain_process_cache(torch)
        info = dict(role=role, rank=int(rank), pid=os.getpid(), limit=limit,
                    allocated=torch.cuda.memory_allocated(device), reserved=torch.cuda.memory_reserved(device),
                    free_before=free, free_after=torch.cuda.mem_get_info(device)[0])
    folder = Path(log_dir)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{role}-rank-{rank}.json"
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(info) + "\n")
    temp.replace(path)
    return info


class SharedCudaReservation:
    def __init__(self, torch_module, device, settings, rank):
        self.torch, self.cuda, self.device = torch_module, torch_module.cuda, device
        self.settings, self.rank = settings, rank
        self.run = os.environ["SPATIAL_WOVR_OUTPUT"]
        self.log = Path(settings["log_dir"]) / f"rank-{rank}.jsonl"
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self.warmed = False
        self.actor_limit = None

    def activate(self):
        self.warmed = True

    def retain(self, phase):
        with (
            self.cuda.device(self.device),
            self.cuda.stream(self.cuda.default_stream(self.device)),
        ):
            self.cuda.synchronize(self.device)
            try:
                snapshot = job_device_memory(self.device, self.run)
            except (
                OSError,
                ValueError,
                ET.ParseError,
                subprocess.SubprocessError,
            ) as error:
                if phase == "initialized":
                    raise
                # A telemetry hiccup must not interrupt a completed update.
                # Leave the already-owned cache intact and record the gap.
                value = {
                    "phase": phase,
                    "time": time.time(),
                    "pid": os.getpid(),
                    "rank": self.rank,
                    "reserved": self.cuda.memory_reserved(self.device),
                    "reservation_error": f"memory query failed: {type(error).__name__}",
                }
                with self.log.open("a") as stream:
                    stream.write(json.dumps(value) + "\n")
                logging.getLogger(__name__).warning(
                    "Shared cache telemetry failed at %s", phase
                )
                return
            full_target = float(self.settings["target_job_gib"])
            target_job = int(
                (
                    full_target
                    if self.warmed
                    else min(
                        full_target,
                        float(self.settings.get("startup_target_job_gib", 65)),
                    )
                )
                * GIB
            )
            headroom = int(float(self.settings["device_headroom_gib"]) * GIB)
            if target_job + headroom > snapshot["device_total"]:
                raise ValueError(
                    "Shared pool target plus headroom exceeds physical VRAM"
                )
            if phase == "initialized" and snapshot["foreign_used"] > 256 * MIB:
                raise RuntimeError(
                    "Shared pool initialization found another GPU job; no training started"
                )
            reserved = self.cuda.memory_reserved(self.device)
            free, total = self.cuda.mem_get_info(self.device)
            peers = self.settings.get("peer_allocator_gib", {})
            if peers and phase == "initialized":
                # All peer workers have returned from initialization. Their reserved
                # pages already count in job_used. Budget for their unfilled cap and
                # for driver/NCCL growth observed at the first weight broadcast.
                growth = 0
                for role, budget in peers.items():
                    path = self.log.parent / f"{role}-rank-{self.rank}.json"
                    info = json.loads(path.read_text())
                    if info["limit"] != int(float(budget) * GIB) or info["rank"] != self.rank:
                        raise ValueError(f"Wrong {role} allocator reservation for rank {self.rank}")
                    growth += max(0, info["limit"] - info["reserved"])
                driver_growth = int(self.settings["driver_growth_gib"] * GIB)
                remaining = min(int(full_target * GIB) - snapshot["job_used"], free - headroom)
                self.actor_limit = (reserved + remaining - growth - driver_growth) // MIB * MIB
                if self.actor_limit <= reserved or self.actor_limit >= total:
                    raise RuntimeError(f"No actor/Wan workspace after peer reservations on rank {self.rank}")
                self.cuda.set_per_process_memory_fraction(self.actor_limit / total, self.device)
            target = shared_reservation_target(
                reserved, free, snapshot["job_used"], target_job, headroom
            )
            if self.actor_limit is not None:
                target = min(target, self.actor_limit - 256 * MIB)
            # Hold small allocations together until the target is reached.
            # Expandable segments coalesce their freed pages on this same stream.
            buffers = []
            reservation_error = None
            try:
                while self.cuda.memory_reserved(self.device) < target:
                    gap = target - self.cuda.memory_reserved(self.device)
                    buffers.append(
                        self.torch.empty(
                            min(256 * MIB, gap),
                            dtype=self.torch.uint8,
                            device=self.device,
                        )
                    )
            except self.torch.OutOfMemoryError as error:
                # Cache top-up is optional after startup. Never turn a failed
                # reservation into a failed optimizer update or empty_cache().
                if phase == "initialized":
                    raise
                reservation_error = str(error)
                logging.getLogger(__name__).warning(
                    "Shared cache top-up failed at %s: %s", phase, error
                )
            finally:
                buffers.clear()
            value = {
                "phase": phase,
                "time": time.time(),
                "pid": os.getpid(),
                "rank": self.rank,
                **snapshot,
                "allocated": self.cuda.memory_allocated(self.device),
                "reserved_before": reserved,
                "allocator_limit": self.actor_limit,
                "peak_allocated": self.cuda.max_memory_allocated(self.device) if self.actor_limit is not None else None,
                "reserved": self.cuda.memory_reserved(self.device),
                "target_reserved": target,
                "target_job": target_job,
                "steady_target_job": int(full_target * GIB),
                "job_used_estimate_after": snapshot["job_used"]
                + self.cuda.memory_reserved(self.device)
                - reserved,
                "free": self.cuda.mem_get_info(self.device)[0],
                "reservation_error": reservation_error,
            }
            with self.log.open("a") as stream:
                stream.write(json.dumps(value) + "\n")
            if (
                phase == "initialized"
                and value["job_used_estimate_after"] < target_job - GIB
            ):
                raise RuntimeError(
                    "Shared allocator could not establish the requested task reservation"
                )


def retain_process_cache(torch_module):
    """Suppress explicit cache release only inside the actor owning Wan.

    Internal allocator OOM recovery remains PyTorch's responsibility. Other
    workers are separate processes and retain their ordinary empty_cache().
    """
    torch_module.cuda.empty_cache = lambda: None
    torch_module.cuda.memory.empty_cache = lambda: None


def start_actor_wan(actor):
    import torch
    from rlinf.envs.world_model.world_model_wan_env import build_wan_pipeline

    settings = actor.cfg.actor.shared_wan_pool
    # RLinf's CUDA platform is torch.cuda: current_device() returns an int.
    # Preserve actor.device for the native actor; normalize only our owner.
    device = (
        torch.device("cuda", actor.device)
        if isinstance(actor.device, int)
        else torch.device(actor.device)
    )
    if torch.cuda.get_allocator_backend() != "native":
        raise ValueError("Shared Wan pool requires native CUDA allocator")
    retain_process_cache(torch)
    reservation = SharedCudaReservation(torch, device, settings, actor._rank)
    actor._cuda_cache_reservation = reservation
    reservation.retain("initialized")
    # Construct on CPU before checkpoint restoration; Wan initialization must
    # not consume the restored policy RNG stream halfway through a rollout.
    with torch.random.fork_rng(devices=[device.index]):
        pipeline = build_wan_pipeline(
            actor.cfg.env.train, str(device), initially_offloaded=True
        )
    pipeline.use_unified_sequence_parallel = False

    @contextmanager
    def execution():
        # The allocator cache is per device and stream. Both threads use the
        # actor's default stream, so cached pages are directly reusable.
        with (
            torch.cuda.device(device),
            torch.cuda.stream(torch.cuda.default_stream(device)),
        ):
            try:
                yield
            finally:
                torch.cuda.synchronize(device)

    owner = SharedWanOwner(
        pipeline,
        device,
        execution=execution,
        retain=reservation.retain,
        activate=reservation.activate,
    )
    actor._shared_wan_owner = owner
    actor._shared_wan_server = SharedWanServer(settings, actor._rank, owner)
    actor.log_info(
        f"Wan now owned by actor rank={actor._rank} pid={os.getpid()}; shared task target={settings.target_job_gib} GiB"
    )
