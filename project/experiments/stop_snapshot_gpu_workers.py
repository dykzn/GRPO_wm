"""Stop verified old GPU workers from an nvidia-smi snapshot, preserving eval."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from stop_spatial_grpo import Process, read_process, signal_process


def snapshot_pids(text: str, suffix: str) -> set[int]:
    """Read only compute process rows whose displayed name matches the target."""
    result = set()
    pattern = re.compile(r"^\|\s*\d+\s+(?:N/A|\d+)\s+(?:N/A|\d+)\s+(\d+)\s+C\s+(.+?)\s+\d+MiB\s*\|\s*$")
    for line in text.splitlines():
        match = pattern.match(line)
        if match and suffix in match[2]:
            result.add(int(match[1]))
    return result


def gpu_processes() -> dict[int, list[str]]:
    command = ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory,gpu_uuid",
               "--format=csv,noheader,nounits"]
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=20)
    rows = {}
    for row in csv.reader(result.stdout.splitlines()):
        if row and row[0].strip().isdecimal():
            rows[int(row[0])] = [value.strip() for value in row[1:]]
    return rows


def boot_time() -> int:
    for line in Path("/proc/stat").read_text().splitlines():
        if line.startswith("btime "):
            return int(line.split()[1])
    raise RuntimeError("Cannot determine process start timestamps")


def rejection_reason(process: Process | None, active_pids: set[int], suffix: str,
                     cutoff: float, boot: int, ticks: int) -> str | None:
    if process is None:
        return "process exited or is invisible in this PID namespace"
    if process.uid != os.geteuid():
        return "owned by a different user"
    if process.pid not in active_pids:
        return "no longer registered as a GPU compute process"
    command = " ".join(process.argv)
    if "wm_eval.py" in command:
        return "protected evaluation process"
    if suffix not in command:
        return "live process name no longer matches the snapshot worker"
    if boot + process.started / ticks > cutoff + 1:
        return "started after the snapshot; possible PID reuse"
    return None


def remaining(targets: dict[int, Process]) -> dict[int, Process]:
    result = {}
    for pid, original in targets.items():
        current = read_process(pid)
        if current and current.uid == os.geteuid() and current.started == original.started:
            result[pid] = current
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", required=True, type=Path)
    parser.add_argument("--worker-suffix", default="PilotSftWorker.run_training")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    path = args.preflight.resolve()
    targets = snapshot_pids(path.read_text(), args.worker_suffix)
    if not targets:
        raise RuntimeError("No matching worker PIDs in this GPU snapshot; nothing was signaled")
    print("Snapshot target PIDs:", ", ".join(map(str, sorted(targets))), flush=True)
    active = gpu_processes()
    boot, ticks, cutoff = boot_time(), os.sysconf("SC_CLK_TCK"), path.stat().st_mtime
    verified = {}
    for pid in sorted(targets):
        process = read_process(pid)
        reason = rejection_reason(process, set(active), args.worker_suffix, cutoff, boot, ticks)
        if reason:
            print(f"Skip PID {pid}: {reason}", flush=True)
        else:
            verified[pid] = process
            print(f"Verified PID {pid}: {' | '.join(active[pid])}", flush=True)
    if not args.apply:
        print("Inspection only. Add --apply to terminate the verified workers.", flush=True)
        return 0
    if not verified:
        print("No verified targets; nothing was signaled. Check this output in the host GPU terminal.", flush=True)
        return 0
    for signum, duration in [(signal.SIGTERM, 10), (signal.SIGKILL, 3)]:
        live = remaining(verified)
        if not live:
            break
        for pid, process in live.items():
            if signal_process(process, signum):
                print(f"{signal.Signals(signum).name}: PID {pid}", flush=True)
        deadline = time.monotonic() + duration
        while remaining(verified) and time.monotonic() < deadline:
            time.sleep(0.5)
    live = remaining(verified)
    if live:
        print(f"Still alive: {sorted(live)}", flush=True)
        return 1
    active = gpu_processes()
    still_registered = targets & set(active)
    if still_registered:
        print(f"Original GPU contexts still registered: {sorted(still_registered)}; recheck nvidia-smi.", flush=True)
        return 1
    print("Verified snapshot workers exited and their GPU contexts disappeared.", flush=True)
    new_workers = []
    for pid in set(active) - targets:
        process = read_process(pid)
        if process and process.uid == os.geteuid() and args.worker_suffix in " ".join(process.argv):
            new_workers.append(pid)
    if new_workers:
        print(f"Other/new workers of the same type remain: {sorted(new_workers)}; their parent job may still be running. They were retained.", flush=True)
    print("Remaining GPU applications (evaluation and other jobs retained):", flush=True)
    for pid, row in sorted(active.items()):
        print(f"PID {pid}: {' | '.join(row)}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        raise SystemExit(f"Cleanup stopped: {exc}. Run in the host GPU terminal.") from exc
