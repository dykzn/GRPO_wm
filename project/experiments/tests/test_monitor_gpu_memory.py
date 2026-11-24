"""Test sampled GPU peaks and monitor lifecycle without requiring a GPU."""

import csv
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from monitor_gpu_memory import Peaks, parse_gpus, parse_processes


def gpu(index, used):
    return {
        "gpu_index": index,
        "gpu_uuid": f"GPU-{index}",
        "name": "Test GPU",
        "total_mib": 81920,
        "used_mib": used,
        "free_mib": 81920 - used,
        "utilization_percent": 50,
    }


def process(pid, used, *, start=1, tracked=True):
    return {
        "gpu_uuid": "GPU-0",
        "pid": pid,
        "pid_start_ticks": start,
        "process_name": "ray::ActorGroup.run_training",
        "used_mib": used,
        "belongs_to_run": tracked,
        "role": "actor",
    }


class ParsingTests(unittest.TestCase):
    def test_device_and_process_csv_support_unavailable_values_and_quoted_names(self):
        rows = parse_gpus(
            "0, GPU-0, NVIDIA A800 80GB PCIe, 81920, 9213, 71941, [N/A]\n"
        )
        self.assertEqual(rows[0]["used_mib"], 9213)
        self.assertIsNone(rows[0]["utilization_percent"])
        rows = parse_processes('GPU-0, 123, "python, worker", [N/A]\n')
        self.assertEqual(rows[0]["process_name"], "python, worker")
        self.assertIsNone(rows[0]["used_mib"])
        self.assertEqual(parse_processes("No running processes found\n"), [])

    def test_invalid_device_data_cannot_be_reported_as_zero_usage(self):
        for text in [
            "",
            "0, GPU-0, A800, 81920, [N/A], 10, 50",
            "0, GPU-0, A800, 81920, -1, 10, 50",
            "0, GPU-0, A800, 81920, nan, 10, 50",
        ]:
            with self.assertRaises(ValueError):
                parse_gpus(text)


class PeakTests(unittest.TestCase):
    def test_device_peak_and_minimum_free_include_other_jobs(self):
        peaks = Peaks()
        peaks.update(
            "first",
            [gpu(0, 10000)],
            [process(1, 1000), process(2, 5000, tracked=False)],
        )
        peaks.update(
            "peak",
            [gpu(0, 50000)],
            [process(1, 35000), process(2, 5000, tracked=False)],
        )
        peaks.update("last", [gpu(0, 20000)], [process(1, 10000)])
        row = peaks.report()["gpus"][0]
        self.assertEqual(row["peak_used_mib"], 50000)
        self.assertEqual(row["peak_timestamp"], "peak")
        self.assertEqual(row["min_free_mib"], 31920)
        self.assertEqual(row["first_sample_used_mib"], 10000)
        self.assertEqual(row["peak_tracked_run_sum_known_mib"], 35000)

    def test_concurrent_peaks_are_not_the_sum_of_individual_maxima(self):
        peaks = Peaks()
        peaks.update("a", [gpu(0, 100), gpu(1, 20)], [process(1, 80), process(2, 10)])
        peaks.update("b", [gpu(0, 20), gpu(1, 100)], [process(1, 10), process(2, 80)])
        report = peaks.report()
        self.assertEqual(report["peak_concurrent_sum_across_selected_gpus_mib"], 120)
        self.assertEqual(report["gpus"][0]["peak_compute_process_sum_known_mib"], 90)
        self.assertEqual(sum(row["peak_used_mib"] for row in report["processes"]), 160)

    def test_unavailable_process_memory_and_pid_reuse_remain_distinct(self):
        peaks = Peaks()
        peaks.update("a", [gpu(0, 100)], [process(1, None, start=1)])
        peaks.update("b", [gpu(0, 200)], [process(1, 100, start=2)])
        rows = peaks.report()["processes"]
        self.assertEqual(len(rows), 2)
        self.assertIsNone(rows[0]["peak_used_mib"])
        self.assertEqual(rows[1]["peak_used_mib"], 100)


class CommandTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.fake = self.root / "fake_nvidia_smi"
        self.fake.write_text(
            f"#!{sys.executable}\n"
            "import sys\n"
            "if any(x.startswith('--query-gpu=') for x in sys.argv):\n"
            " print('0, GPU-0, Fake A800, 81920, 12000, 69920, 25')\n"
            "else:\n"
            " print('GPU-0, 99999999, ray::ActorGroup.run_training, 8000')\n"
        )
        self.fake.chmod(0o755)
        self.output = self.root / "output"
        self.command = [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "monitor_gpu_memory.py"),
            "--gpus",
            "0",
            "--interval",
            "0.05",
            "--print-every",
            "0.1",
            "--nvidia-smi",
            str(self.fake),
            "--output",
            str(self.output),
        ]

    def tearDown(self):
        self.temporary.cleanup()

    def test_duration_saves_live_csv_and_final_summary(self):
        result = subprocess.run(
            [*self.command, "--duration", "0.2"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads((self.output / "summary.json").read_text())
        self.assertEqual(report["stop_reason"], "duration_reached")
        self.assertGreaterEqual(report["samples"], 2)
        self.assertEqual(report["gpus"][0]["peak_used_mib"], 12000)
        with (self.output / "gpu_samples.csv").open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), report["samples"])
        with (self.output / "process_samples.csv").open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), report["samples"])

    def test_ctrl_c_writes_summary_and_exits_cleanly(self):
        child = subprocess.Popen(
            self.command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        try:
            deadline = time.monotonic() + 5
            while (
                not (self.output / "summary.json").exists()
                and time.monotonic() < deadline
            ):
                if child.poll() is not None:
                    self.fail("Monitor exited before its first summary")
                time.sleep(0.02)
            self.assertTrue((self.output / "summary.json").exists())
            os.kill(child.pid, signal.SIGINT)
            _, error = child.communicate(timeout=5)
            self.assertEqual(child.returncode, 0, error)
            report = json.loads((self.output / "summary.json").read_text())
            self.assertEqual(report["stop_reason"], "SIGINT")
            self.assertGreaterEqual(report["samples"], 1)
        finally:
            if child.poll() is None:
                child.terminate()
                child.communicate(timeout=5)

    def test_missing_driver_fails_without_fabricating_measurements(self):
        missing = self.root / "missing_nvidia_smi"
        command = list(self.command)
        command[command.index(str(self.fake))] = str(missing)
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=5, check=False
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("GPU terminal", result.stderr)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
