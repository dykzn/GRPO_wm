"""Sample physical GPU/process memory and save observed peaks without CUDA.

Run this in the GPU terminal, before or during training. Ctrl+C stops only
the monitor and writes its final summary. No training processes are signaled.
This records sampled driver usage, including CUDA caches and other jobs;
allocations shorter than the sampling interval may not be observed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def number(value: str) -> float | None:
    value = value.strip()
    if value in {"N/A", "[N/A]", "Not Supported", "[Not Supported]", ""}:
        return None
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"Invalid GPU counter: {value!r}")
    return result


def parse_gpus(output: str) -> list[dict]:
    rows = []
    for values in csv.reader(output.splitlines(), skipinitialspace=True):
        if not values or not any(value.strip() for value in values):
            continue
        if len(values) != 7:
            raise ValueError(f"Expected 7 GPU fields, got {len(values)}")
        index, uuid, name, total, used, free, utilization = map(str.strip, values)
        counters = [number(value) for value in (total, used, free)]
        if any(value is None for value in counters):
            raise ValueError(f"GPU {index} has unavailable device-memory counters")
        rows.append(
            {
                "gpu_index": int(index),
                "gpu_uuid": uuid,
                "name": name,
                "total_mib": counters[0],
                "used_mib": counters[1],
                "free_mib": counters[2],
                "utilization_percent": number(utilization),
            }
        )
    if not rows or len({row["gpu_index"] for row in rows}) != len(rows):
        raise ValueError("No GPUs or duplicate GPU indices in nvidia-smi output")
    return rows


def parse_processes(output: str) -> list[dict]:
    rows = []
    for values in csv.reader(output.splitlines(), skipinitialspace=True):
        if not values or not any(value.strip() for value in values):
            continue
        if len(values) == 1 and values[0].strip().startswith("No running processes"):
            continue
        if len(values) != 4:
            raise ValueError(f"Expected 4 compute-process fields, got {len(values)}")
        uuid, pid, name, memory = map(str.strip, values)
        rows.append(
            {
                "gpu_uuid": uuid,
                "pid": int(pid),
                "process_name": name,
                "used_mib": number(memory),
            }
        )
    return rows


class NvidiaSMI:
    def __init__(self, executable: str):
        self.executable = executable

    def query(self, kind: str, fields: str) -> str:
        command = [
            self.executable,
            f"--query-{kind}={fields}",
            "--format=csv,noheader,nounits",
        ]
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=10, check=False
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "nvidia-smi was not found; run in the GPU terminal"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("nvidia-smi query timed out") from exc
        if result.returncode:
            message = (result.stderr or result.stdout).strip()[:1000]
            raise RuntimeError(f"nvidia-smi failed ({result.returncode}): {message}")
        return result.stdout

    def sample(self) -> tuple[list[dict], list[dict], str | None]:
        gpus = parse_gpus(
            self.query(
                "gpu",
                "index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu",
            )
        )
        try:
            processes = parse_processes(
                self.query("compute-apps", "gpu_uuid,pid,process_name,used_gpu_memory")
            )
            process_error = None
        except (RuntimeError, ValueError) as exc:
            processes, process_error = [], str(exc)
        return gpus, processes, process_error


def process_identity(pid: int, process_name: str, run_dir: Path | None) -> dict:
    """Read identity/role without storing command lines or environment secrets."""
    identity = {"pid_start_ticks": None, "role": "unknown", "belongs_to_run": None}
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        identity["pid_start_ticks"] = int(stat.rsplit(") ", 1)[1].split()[19])
        command = (
            Path(f"/proc/{pid}/cmdline")
            .read_bytes()
            .replace(b"\0", b" ")
            .decode(errors="replace")
        )
    except (OSError, ValueError, IndexError):
        command = ""
    description = f"{process_name} {command}".lower()
    for role, keyword in [
        ("cosmos", "cosmos_server"),
        ("actor", "actorgroup"),
        ("rollout", "rolloutgroup"),
        ("env", "envgroup"),
    ]:
        if keyword in description:
            identity["role"] = role
            break
    if run_dir is not None:
        try:
            entries = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
            prefix = b"SPATIAL_WOVR_OUTPUT="
            value = next(
                (entry[len(prefix) :] for entry in entries if entry.startswith(prefix)),
                None,
            )
            if value is not None:
                identity["belongs_to_run"] = (
                    Path(os.fsdecode(value)).resolve() == run_dir
                )
        except OSError:
            pass
    return identity


class Peaks:
    """Aggregate concurrent samples; never sum peaks taken at different times."""

    def __init__(self):
        self.gpus = {}
        self.processes = {}
        self.samples = 0
        self.aggregate_peak_mib = 0.0

    def update(self, timestamp: str, gpus: list[dict], processes: list[dict]) -> None:
        self.samples += 1
        self.aggregate_peak_mib = max(
            self.aggregate_peak_mib, sum(gpu["used_mib"] for gpu in gpus)
        )
        for gpu in gpus:
            uuid = gpu["gpu_uuid"]
            active = [row for row in processes if row["gpu_uuid"] == uuid]
            tracked = [row for row in active if row.get("belongs_to_run") is True]
            state = self.gpus.setdefault(
                uuid,
                {
                    "gpu_index": gpu["gpu_index"],
                    "gpu_uuid": uuid,
                    "name": gpu["name"],
                    "total_mib": gpu["total_mib"],
                    "first_sample_used_mib": gpu["used_mib"],
                    "peak_used_mib": -1.0,
                    "min_free_mib": gpu["free_mib"],
                    "peak_timestamp": timestamp,
                    "peak_compute_process_sum_known_mib": None,
                    "peak_tracked_run_sum_known_mib": None,
                    "processes_at_device_peak": [],
                },
            )
            state["min_free_mib"] = min(state["min_free_mib"], gpu["free_mib"])
            if gpu["used_mib"] > state["peak_used_mib"]:
                state.update(
                    peak_used_mib=gpu["used_mib"],
                    peak_timestamp=timestamp,
                    processes_at_device_peak=active,
                )
            if active:
                value = sum(
                    row["used_mib"] for row in active if row["used_mib"] is not None
                )
                previous = state["peak_compute_process_sum_known_mib"]
                state["peak_compute_process_sum_known_mib"] = max(
                    value, previous or 0.0
                )
            if tracked:
                value = sum(
                    row["used_mib"] for row in tracked if row["used_mib"] is not None
                )
                previous = state["peak_tracked_run_sum_known_mib"]
                state["peak_tracked_run_sum_known_mib"] = max(value, previous or 0.0)
        for process in processes:
            key = (process["gpu_uuid"], process["pid"], process.get("pid_start_ticks"))
            state = self.processes.setdefault(
                key, dict(process, peak_used_mib=None, peak_timestamp=None)
            )
            used = process["used_mib"]
            if used is not None and (
                state["peak_used_mib"] is None or used > state["peak_used_mib"]
            ):
                state.update(process, peak_used_mib=used, peak_timestamp=timestamp)

    def report(self, **metadata) -> dict:
        gpus = []
        for state in sorted(self.gpus.values(), key=lambda row: row["gpu_index"]):
            row = dict(state)
            row["peak_used_gib"] = row["peak_used_mib"] / 1024
            row["peak_minus_first_sample_mib"] = (
                row["peak_used_mib"] - row["first_sample_used_mib"]
            )
            gpus.append(row)
        return dict(
            **metadata,
            samples=self.samples,
            gpus=gpus,
            processes=list(self.processes.values()),
            peak_concurrent_sum_across_selected_gpus_mib=self.aggregate_peak_mib,
            interpretation=[
                "Peaks are observed samples, not exact allocator high-water marks.",
                "Device usage includes every job, CUDA caches, graphics and driver memory.",
                "Per-process entries cover compute apps only; unavailable memory remains null.",
                "Known process sums are lower bounds if any process memory is unavailable.",
                "Run attribution requires readable /proc and inherited SPATIAL_WOVR_OUTPUT.",
                "Peak minus first sample is a device change, not isolated training memory.",
                "GPU and process queries are sequential and are not an atomic snapshot.",
            ],
        )


def write_summary(output: Path, report: dict) -> None:
    temporary = output / "summary.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / "summary.json")
    fields = [
        "gpu_index",
        "name",
        "first_sample_used_mib",
        "peak_used_mib",
        "peak_used_gib",
        "min_free_mib",
        "peak_timestamp",
        "peak_compute_process_sum_known_mib",
        "peak_tracked_run_sum_known_mib",
    ]
    temporary = output / "gpu_peaks.csv.tmp"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["gpus"])
    temporary.replace(output / "gpu_peaks.csv")


def display(peaks: Peaks, seconds: float) -> None:
    text = " | ".join(
        f"GPU {row['gpu_index']}: {row['peak_used_mib'] / 1024:.2f} GiB"
        for row in sorted(peaks.gpus.values(), key=lambda row: row["gpu_index"])
    )
    print(
        f"[{seconds:.1f}s, {peaks.samples} samples] observed device peaks: {text}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpus", default="0,1,2,3,4,5,6,7", help="Physical GPU indices, or all"
    )
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument(
        "--duration", type=float, default=0, help="Seconds; 0 monitors until Ctrl+C"
    )
    parser.add_argument("--print-every", type=float, default=10)
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Optional existing SPATIAL_WOVR_OUTPUT for PID attribution",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    args = parser.parse_args()
    if (
        not all(
            math.isfinite(value)
            for value in (args.interval, args.duration, args.print_every)
        )
        or args.interval <= 0
        or args.duration < 0
        or args.print_every <= 0
    ):
        parser.error(
            "Interval/print period must be positive; duration must be nonnegative"
        )
    selected = None
    if args.gpus != "all":
        parts = args.gpus.split(",")
        if any(not part.strip().isdecimal() for part in parts):
            parser.error("--gpus must contain physical numeric indices, or all")
        selected = {int(part) for part in parts}
        if len(selected) != len(parts):
            parser.error("Duplicate GPU indices")
    if args.run_dir is not None:
        args.run_dir = args.run_dir.resolve()
        if not args.run_dir.is_dir():
            parser.error("--run-dir must be an existing training output directory")
    source = NvidiaSMI(args.nvidia_smi)
    first = (
        source.sample()
    )  # Fail before creating outputs when CUDA/driver is invisible.
    available = {row["gpu_index"] for row in first[0]}
    if selected is not None and not selected <= available:
        raise RuntimeError(
            f"Requested unavailable physical GPUs: {sorted(selected - available)}"
        )
    output = (
        args.output
        or Path(__file__).resolve().parents[1]
        / "runtime_artifacts/gpu_memory"
        / f"{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{os.getpid()}"
    )
    output.mkdir(parents=True, exist_ok=True)
    if any(
        (output / name).exists()
        for name in ("summary.json", "gpu_samples.csv", "process_samples.csv")
    ):
        raise RuntimeError(f"Output already contains a monitoring run: {output}")
    reason = None
    stopped = threading.Event()

    def stop(signum, frame):
        nonlocal reason
        reason = signal.Signals(signum).name
        stopped.set()

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, stop)
    started, started_utc = time.monotonic(), utc_now()
    peaks, errors = Peaks(), []
    identity_cache = {}
    next_print = 0.0
    print(
        f"GPU memory monitor: {output}\nInterval={args.interval}s. Ctrl+C stops only this monitor.",
        flush=True,
    )

    def report():
        return peaks.report(
            started_utc=started_utc,
            updated_utc=utc_now(),
            elapsed_seconds=time.monotonic() - started,
            requested_interval_seconds=args.interval,
            physical_gpu_indices=sorted(selected or available),
            run_dir=str(args.run_dir) if args.run_dir else None,
            stop_reason=reason or "running",
            query_errors=errors,
        )

    gpu_fields = [
        "timestamp_utc",
        "elapsed_seconds",
        "gpu_index",
        "gpu_uuid",
        "name",
        "total_mib",
        "used_mib",
        "free_mib",
        "utilization_percent",
    ]
    process_fields = [
        "timestamp_utc",
        "elapsed_seconds",
        "gpu_index",
        "gpu_uuid",
        "pid",
        "pid_start_ticks",
        "process_name",
        "role",
        "belongs_to_run",
        "used_mib",
    ]
    with (
        (output / "gpu_samples.csv").open("w", newline="") as gpu_file,
        (output / "process_samples.csv").open("w", newline="") as process_file,
    ):
        gpu_writer = csv.DictWriter(gpu_file, fieldnames=gpu_fields)
        process_writer = csv.DictWriter(process_file, fieldnames=process_fields)
        gpu_writer.writeheader()
        process_writer.writeheader()
        failures = 0
        try:
            while reason is None:
                sample_started = time.monotonic()
                try:
                    gpus, processes, process_error = (
                        first if first is not None else source.sample()
                    )
                    first = None
                    failures = 0
                except (RuntimeError, ValueError) as exc:
                    errors.append({"timestamp_utc": utc_now(), "error": str(exc)})
                    failures += 1
                    if failures >= 3:
                        reason = "three_consecutive_gpu_query_failures"
                        break
                    stopped.wait(min(args.interval, 1))
                    continue
                timestamp, elapsed = utc_now(), time.monotonic() - started
                if process_error:
                    errors.append({"timestamp_utc": timestamp, "error": process_error})
                gpus = [
                    row
                    for row in gpus
                    if selected is None or row["gpu_index"] in selected
                ]
                gpu_indices = {row["gpu_uuid"]: row["gpu_index"] for row in gpus}
                active_pids = {row["pid"] for row in processes}
                identity_cache = {
                    pid: value
                    for pid, value in identity_cache.items()
                    if pid in active_pids
                }
                processes = [row for row in processes if row["gpu_uuid"] in gpu_indices]
                for row in processes:
                    cached = identity_cache.get(row["pid"])
                    if cached is None or elapsed - cached[0] >= 5:
                        cached = (
                            elapsed,
                            process_identity(
                                row["pid"], row["process_name"], args.run_dir
                            ),
                        )
                        identity_cache[row["pid"]] = cached
                    row.update(cached[1], gpu_index=gpu_indices[row["gpu_uuid"]])
                    process_writer.writerow(
                        dict(timestamp_utc=timestamp, elapsed_seconds=elapsed, **row)
                    )
                for row in gpus:
                    gpu_writer.writerow(
                        dict(timestamp_utc=timestamp, elapsed_seconds=elapsed, **row)
                    )
                peaks.update(timestamp, gpus, processes)
                if elapsed >= next_print:
                    gpu_file.flush()
                    process_file.flush()
                    write_summary(output, report())
                    display(peaks, elapsed)
                    next_print = elapsed + args.print_every
                if args.duration and time.monotonic() - started >= args.duration:
                    reason = "duration_reached"
                    break
                remaining = args.interval - (time.monotonic() - sample_started)
                if args.duration:
                    remaining = min(
                        remaining, args.duration - (time.monotonic() - started)
                    )
                if remaining > 0 and reason is None:
                    stopped.wait(remaining)
        finally:
            if reason is None:
                reason = "unexpected_error"
            write_summary(output, report())
            display(peaks, time.monotonic() - started)
            print(
                f"Summary: {output / 'summary.json'}\nTable: {output / 'gpu_peaks.csv'}",
                flush=True,
            )
    return (
        1
        if reason in {"unexpected_error", "three_consecutive_gpu_query_failures"}
        else 0
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
