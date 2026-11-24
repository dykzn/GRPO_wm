"""Inspect or stop processes belonging to one simulator Spatial GRPO run."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import re
import signal
import time


@dataclass(frozen=True)
class Process:
    pid: int
    ppid: int
    uid: int
    started: int
    name: str
    argv: tuple[str, ...]
    environment: dict[str, str]


def read_process(pid: int, proc_root: Path = Path("/proc")) -> Process | None:
    """Read identity twice so a reused PID cannot mix two process records."""
    root = proc_root / str(pid)
    try:
        fields = (root / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        started = int(fields[19])
        uid = root.stat().st_uid
        argv = tuple(part.decode(errors="replace") for part in (root / "cmdline").read_bytes().split(b"\0") if part)
        name = (root / "comm").read_text().strip()
        try:
            raw_environment = (root / "environ").read_bytes()
        except PermissionError:
            raw_environment = b""
        environment = {}
        for part in raw_environment.split(b"\0"):
            key, separator, value = part.partition(b"=")
            if separator and key in {b"SPATIAL_GRPO_OUTPUT", b"RAY_TMPDIR", b"TMPDIR"}:
                environment[key.decode()] = value.decode(errors="replace")
        again = (root / "stat").read_text().rsplit(")", 1)[1].split()
        if int(again[19]) != started or again[0] == "Z":
            return None
        return Process(pid, int(fields[1]), uid, started, name, argv, environment)
    except (OSError, ValueError, IndexError):
        return None


def snapshot() -> dict[int, Process]:
    processes = {}
    for entry in Path("/proc").iterdir():
        if entry.name.isdecimal():
            process = read_process(int(entry.name))
            if process is not None and process.uid == os.geteuid():
                processes[process.pid] = process
    return processes


def same_path(value: str, expected: Path) -> bool:
    return bool(value) and Path(value).is_absolute() and Path(value).resolve() == expected


def ray_roots(run: Path) -> set[str]:
    """Recover this launcher's unique Ray directory from its existing log."""
    roots = set()
    pattern = re.compile(r"/tmp/spatial-grpo8-[A-Za-z0-9]+(?=/|\s|$)")
    with (run / "train.log").open(errors="replace") as stream:
        for line in stream:
            roots.update(pattern.findall(line))
    return roots


def protected_ancestors(processes: dict[int, Process], pid: int) -> set[int]:
    protected = {pid}
    while pid in processes and processes[pid].ppid not in protected:
        pid = processes[pid].ppid
        protected.add(pid)
    return protected


def select_processes(processes: dict[int, Process], run: Path, roots: set[str],
                     excluded: set[int], tracked: dict[int, int] | None = None) -> dict[int, str]:
    """Use run-specific environment/Ray paths, then include their descendants."""
    selected = {}
    allowed = {}
    for pid, process in processes.items():
        if pid in excluded or process.uid != os.geteuid():
            continue
        output = process.environment.get("SPATIAL_GRPO_OUTPUT", "")
        # An explicitly different run is protected even if it is a descendant.
        if output and not same_path(output, run):
            continue
        allowed[pid] = process
        if tracked and tracked.get(pid) == process.started:
            selected[pid] = "previously identified process"
        elif output:
            selected[pid] = "SPATIAL_GRPO_OUTPUT matches run"
        elif any(process.environment.get(key) in roots for key in ["RAY_TMPDIR", "TMPDIR"]):
            selected[pid] = "private Ray temporary directory matches"
        else:
            ray_script = process.name.startswith("python") and any(
                argument.endswith(".py") and any(marker in argument for marker in
                    ["/ray/_private/", "/ray/dashboard/", "/ray/autoscaler/"])
                for argument in process.argv)
            ray_service = (process.name in {"raylet", "gcs_server"}
                           or process.name.startswith("ray::") or ray_script)
            if ray_service and any(re.search(re.escape(root) + r"(?=/|\s|$)", argument)
                                   for root in roots for argument in process.argv):
                selected[pid] = "private Ray service path matches"
    while True:
        children = {pid: f"child of identified PID {process.ppid}" for pid, process in allowed.items()
                    if pid not in selected and process.ppid in selected}
        if not children:
            return selected
        selected.update(children)


def signal_process(process: Process, signum: int) -> bool:
    """Recheck owner and start time; use a pidfd when the kernel supports it."""
    descriptor = None
    try:
        if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
            try:
                descriptor = os.pidfd_open(process.pid)
            except OSError as exc:
                if exc.errno not in {22, 38}:  # Older kernels: EINVAL/ENOSYS.
                    raise
        current = read_process(process.pid)
        if current is None or current.uid != os.geteuid() or current.started != process.started:
            return False
        if descriptor is not None:
            signal.pidfd_send_signal(descriptor, signum)
        else:
            os.kill(process.pid, signum)
        return True
    except ProcessLookupError:
        return False
    finally:
        if descriptor is not None:
            os.close(descriptor)


def stop(run: Path, roots: set[str], initial: dict[int, Process], excluded: set[int], grace: float) -> int:
    tracked = {pid: process.started for pid, process in initial.items()}
    sent = set()
    for signum, duration in [(signal.SIGTERM, grace), (signal.SIGKILL, 3.0)]:
        deadline = time.monotonic() + duration
        while True:
            processes = snapshot()
            selected = select_processes(processes, run, roots, excluded, tracked)
            if not selected:
                print("No matching live GRPO processes remain in this PID namespace.", flush=True)
                return 0
            for pid in selected:
                process = processes[pid]
                tracked[pid] = process.started
                identity = (pid, process.started, signum)
                if identity not in sent:
                    if signal_process(process, signum):
                        print(f"{signal.Signals(signum).name}: PID {pid} {process.name}", flush=True)
                    sent.add(identity)
            if time.monotonic() >= deadline:
                break
            time.sleep(0.5)
    processes = snapshot()
    remaining = select_processes(processes, run, roots, excluded, tracked)
    if remaining:
        print(f"Still alive: {sorted(remaining)}; inspect these PIDs in the GPU terminal.", flush=True)
        return 1
    print("No matching live GRPO processes remain in this PID namespace.", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--apply", action="store_true", help="Send TERM, then KILL only to verified processes of this run")
    parser.add_argument("--grace", type=float, default=10.0)
    args = parser.parse_args()
    if not 0 <= args.grace <= 60:
        parser.error("--grace must be between 0 and 60 seconds")
    run = args.run_dir.resolve()
    for name in ["train.log", "resolved_config.yaml", "train_spatial_grpo.py"]:
        if not (run / name).is_file():
            parser.error(f"Not a recognized simulator GRPO run: missing {run / name}")
    roots = ray_roots(run)
    processes = snapshot()
    excluded = protected_ancestors(processes, os.getpid())
    selected = select_processes(processes, run, roots, excluded)
    print(f"Run: {run}")
    print(f"Private Ray directories: {', '.join(sorted(roots)) or 'not found in log'}")
    for pid in sorted(selected):
        print(f"PID {pid:<9} {processes[pid].name:<28} {selected[pid]}")
    if not selected:
        print("No matching processes visible. This does not establish that host GPU memory is free.")
        print("If nvidia-smi still lists GRPO workers, run this script in that same host/GPU terminal.")
        return 0
    if not args.apply:
        print("Inspection only. Add --apply to stop these processes. Files are retained.")
        return 0
    return stop(run, roots, {pid: processes[pid] for pid in selected}, excluded, args.grace)


if __name__ == "__main__":
    raise SystemExit(main())
