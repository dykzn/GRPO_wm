"""Exercise contention, failed inference, and committed checkpoint recovery."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_spatial_wovr_resilient_8gpu as supervisor
from spatial_wovr_memory_guard import (
    GuardSettings,
    MemoryGuard,
    MemoryGuardRunnerMixin,
    checkpoint_complete,
    configure_recovery,
    query_snapshot,
)


def make_checkpoint(path, *, committed=True, step=None):
    path = Path(path)
    files = ["actor/model_state_dict/full_weights.pt", "actor/dcp_checkpoint/.metadata"]
    files += [f"actor/dcp_checkpoint/__{rank}_0.distcp" for rank in range(8)]
    for name in files:
        file = path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b"complete fixture")
    if committed:
        (path / "recovery_complete.json").write_text(
            json.dumps(
                {
                    "step": step
                    if step is not None
                    else supervisor.checkpoint_step(path),
                    "world_size": 8,
                }
            )
        )
    return path


def snapshot(foreign=0, free=45000):
    return [{"index": 0, "foreign_mib": foreign, "free_mib": free, "foreign": []}]


class ContentionTests(unittest.TestCase):
    def guard(self, probes):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        probe = Mock(side_effect=probes)
        guard = MemoryGuard(
            GuardSettings(enabled=True), [0], temp.name, probe=probe, sleep=Mock()
        )
        return guard, probe

    def test_every_device_must_have_room_and_queries_fail_closed(self):
        guard, probe = self.guard(
            [
                snapshot(foreign=22000),
                subprocess.TimeoutExpired("nvidia-smi", 15),
                snapshot(),
                snapshot(free=1000),
                snapshot(),
                snapshot(),
            ]
        )
        self.assertTrue(guard.wait("admission", required_free_gib=38, force=True))
        self.assertEqual(probe.call_count, 6)
        self.assertEqual(guard.sleep.call_count, 5)

    def test_one_blocked_card_holds_admission_for_all_cards(self):
        good = snapshot()[0]
        bad = snapshot(foreign=22000)[0]
        guard, probe = self.guard([[good, bad], [good, good], [good, good]])
        guard.wait("all_ranks", required_free_gib=12)
        self.assertEqual(probe.call_count, 3)

    def test_startup_requires_three_consecutive_idle_observations(self):
        guard, probe = self.guard(
            [
                snapshot(),
                snapshot(),
                snapshot(foreign=3000),
                snapshot(),
                snapshot(),
                snapshot(),
            ]
        )
        guard.settings = GuardSettings(enabled=True, stable_checks=3)
        guard.wait("startup", require_stable=True)
        self.assertEqual(probe.call_count, 6)

    def test_inference_oom_waits_and_repeats_exact_inputs(self):
        guard, probe = self.guard(
            [snapshot(), snapshot(foreign=22000), snapshot(), snapshot()]
        )
        received = []
        state = {"history": 0}

        def predict():
            received.append((123, (1, 2, 3)))
            if len(received) == 1:
                raise RuntimeError("CUDA out of memory. synthetic")
            return "future"

        release = Mock()
        result = guard.inference(predict, phase="wan", release_cache=release)
        state["history"] += (
            1  # commit follows successful inference, never a failed attempt
        )
        self.assertEqual(result, "future")
        self.assertEqual(received, [(123, (1, 2, 3))] * 2)
        self.assertEqual(state["history"], 1)
        release.assert_called_once_with()
        self.assertEqual(probe.call_count, 4)

    def test_unrelated_errors_propagate_without_repeating_calls(self):
        guard, _ = self.guard([snapshot()])
        predict = Mock(side_effect=ValueError("wrong action shape"))
        with self.assertRaisesRegex(ValueError, "wrong action"):
            guard.inference(predict, phase="wrist")
        predict.assert_called_once_with()

    def test_persistent_oom_without_contention_is_not_hidden_forever(self):
        guard, _ = self.guard([snapshot()] * 3)
        predict = Mock(side_effect=RuntimeError("CUDA out of memory"))
        with self.assertRaisesRegex(RuntimeError, "without observable contention"):
            guard.inference(predict, phase="wan")
        self.assertEqual(predict.call_count, 2)

    def test_query_excludes_only_verified_members_of_the_same_run(self):
        responses = [
            "0, GPU-a, 40000\n1, GPU-b, 41000\n",
            "GPU-a, 101, 10000\nGPU-a, 202, 22000\nGPU-b, 303, 5000\n",
        ]
        command = Mock(
            side_effect=[
                subprocess.CompletedProcess([], 0, stdout=x) for x in responses
            ]
        )
        rows = query_snapshot(
            [0], "/run/current", command=command, owns=lambda pid, run: pid == 101
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["foreign_mib"], 22000)
        self.assertEqual(rows[0]["foreign"], [{"pid": 202, "used_mib": 22000}])

    def test_missing_device_is_an_error_not_an_empty_success(self):
        command = Mock(
            return_value=subprocess.CompletedProcess([], 0, stdout="0, GPU-a, 40000\n")
        )
        with self.assertRaisesRegex(ValueError, "every requested"):
            query_snapshot([0, 1], "/run/current", command=command)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.initial = make_checkpoint(self.base / "original/global_step_10")

    def test_partial_and_uncommitted_newer_checkpoints_are_skipped(self):
        root = self.base / "runs"
        valid = make_checkpoint(
            root / "attempt_001_from_10/experiment/checkpoints/global_step_11"
        )
        make_checkpoint(
            root / "attempt_001_from_10/experiment/checkpoints/global_step_12",
            committed=False,
        )
        partial = make_checkpoint(
            root / "attempt_002_from_11/experiment/checkpoints/global_step_13"
        )
        (partial / "actor/dcp_checkpoint/__7_0.distcp").unlink()
        self.assertEqual(supervisor.latest_recovery(root, self.initial), valid)

    def test_marker_with_wrong_step_is_not_trusted(self):
        root = self.base / "runs"
        make_checkpoint(
            root / "attempt_001_from_10/experiment/checkpoints/global_step_20", step=999
        )
        self.assertEqual(supervisor.latest_recovery(root, self.initial), self.initial)

    def test_complete_requires_all_eight_nonempty_optimizer_shards(self):
        self.assertTrue(checkpoint_complete(self.initial, 8))
        (self.initial / "actor/dcp_checkpoint/__4_0.distcp").write_bytes(b"")
        self.assertFalse(checkpoint_complete(self.initial, 8))

    def test_recovery_config_covers_every_step_while_preserving_eval_schedule(self):
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(
            {
                "runner": {
                    "max_steps": 20,
                    "val_check_interval": 5,
                    "save_interval": 5,
                    "memory_guard": {"enabled": True},
                },
                "actor": {"fsdp_config": {"save_training_state_at_steps": [10, 20]}},
            }
        )
        configure_recovery(cfg)
        self.assertEqual(
            list(cfg.actor.fsdp_config.save_training_state_at_steps), list(range(1, 21))
        )
        self.assertEqual(cfg.runner.val_check_interval, 5)
        self.assertEqual(cfg.runner.save_interval, 5)

    def test_service_startup_oom_is_retryable_without_a_train_log(self):
        run = self.base / "run"
        run.mkdir()
        (run / "cosmos_rank3.log").write_text(
            "torch.OutOfMemoryError: CUDA out of memory"
        )
        self.assertTrue(supervisor.failed_with_oom(run))

    def test_non_memory_failure_is_not_classified_from_recovery_warning(self):
        run = self.base / "run"
        run.mkdir()
        (run / "train.log").write_text(
            "[memory-guard] inference OOM phase=wan; checking contention\nValueError: malformed data"
        )
        self.assertFalse(supervisor.failed_with_oom(run))

    def test_supervisor_restarts_from_last_committed_step_after_oom(self):
        calls = []

        def start(command, **options):
            output = Path(options["env"]["SPATIAL_WOVR_OUTPUT"])
            output.mkdir(parents=True)
            calls.append(command)
            number = len(calls)
            step = 11 if number == 1 else 20
            make_checkpoint(output / f"experiment/checkpoints/global_step_{step}")
            if number == 1:
                make_checkpoint(
                    output / "experiment/checkpoints/global_step_12", committed=False
                )
                (output / "train.log").write_text(
                    "torch.OutOfMemoryError: CUDA out of memory"
                )
            code = 1 if number == 1 else 0
            return SimpleNamespace(pid=100 + number, returncode=code, poll=lambda: code)

        root = self.base / "supervised"
        with (
            patch.object(supervisor.subprocess, "Popen", side_effect=start),
            patch.object(supervisor.MemoryGuard, "wait", return_value=False),
            patch.object(supervisor.time, "sleep"),
            patch.object(supervisor.signal, "signal"),
        ):
            self.assertEqual(supervisor.supervise(root, self.initial, 20), 0)
        self.assertEqual(len(calls), 2)
        resume = next(x for x in calls[1] if x.startswith("runner.resume_dir="))
        self.assertTrue(resume.endswith("global_step_11"), resume)
        self.assertEqual(
            json.loads((root / "supervisor_state.json").read_text())["status"],
            "complete",
        )


class CheckpointMixinTests(unittest.TestCase):
    def test_recovery_marker_is_written_only_after_distributed_save_returns(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = SimpleNamespace(
                runner=SimpleNamespace(
                    logger=SimpleNamespace(
                        log_path=folder, experiment_name="experiment"
                    ),
                    save_interval=5,
                )
            )

            class Base:
                def _maybe_eval_and_checkpoint(self, step):
                    return {"eval": 0.5}

            class Runner(MemoryGuardRunnerMixin, Base):
                def _memory_guard(self):
                    return SimpleNamespace(
                        settings=GuardSettings(enabled=True), devices=list(range(8))
                    )

                def _save_checkpoint(self):
                    path = Path(folder) / "experiment/checkpoints/global_step_11"
                    self.assert_not_marked = not (
                        path / "recovery_complete.json"
                    ).exists()
                    make_checkpoint(path, committed=False)

            runner = Runner()
            runner.cfg, runner.global_step, runner.max_steps = cfg, 11, 20
            self.assertEqual(runner._maybe_eval_and_checkpoint(10), {"eval": 0.5})
            self.assertTrue(runner.assert_not_marked)
            marker = (
                Path(folder)
                / "experiment/checkpoints/global_step_11/recovery_complete.json"
            )
            self.assertEqual(json.loads(marker.read_text())["step"], 11)


if __name__ == "__main__":
    unittest.main()
