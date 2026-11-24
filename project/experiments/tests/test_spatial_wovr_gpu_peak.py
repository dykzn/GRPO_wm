"""Test exact allocator measurements and the native actor hook boundary."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spatial_wovr_gpu_peak import GIB, ActorUpdatePeak, write_peak_summary
from spatial_wovr_launch import validate_batch


class FakeCuda:
    def __init__(self):
        self.allocated = GIB
        self.reserved = 8 * GIB
        self.peak_allocated = 8 * GIB  # Earlier memory-pool warmup.
        self.peak_reserved = self.reserved

    def synchronize(self, device):
        pass

    def reset_peak_memory_stats(self, device):
        self.peak_allocated = self.allocated
        self.peak_reserved = self.reserved

    def memory_allocated(self, device):
        return self.allocated

    def memory_reserved(self, device):
        return self.reserved

    def max_memory_allocated(self, device):
        return self.peak_allocated

    def max_memory_reserved(self, device):
        return self.peak_reserved

    def mem_get_info(self, device):
        return 70 * GIB, 80 * GIB


def actor(output):
    cfg = OmegaConf.create(
        {
            "actor": {
                "micro_batch_size": 32,
                "gpu_memory_profile": {
                    "enabled": True,
                    "expected_optimizer_updates": 1,
                },
            },
            "runner": {"logger": {"log_path": str(output)}},
        }
    )
    return SimpleNamespace(
        cfg=cfg,
        device=0,
        _rank=0,
        version=0,
        optimizer_steps=0,
        gradient_accumulation=8,
    )


class PeakProfileTests(unittest.TestCase):
    def test_four_gpu_accumulation_counts_sixteen_backwards_for_one_update(self):
        with tempfile.TemporaryDirectory() as output:
            instance, cuda = actor(output), FakeCuda()
            instance.gradient_accumulation = 16
            profile = ActorUpdatePeak(instance, SimpleNamespace(cuda=cuda))
            profile.start()
            for _ in range(16):
                profile.after_backward()
            instance.optimizer_steps = 1
            report = profile.finish(metrics={"actor/grad_norm": 0.5})
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["backward_calls_completed"], 16)

    def test_four_gpu_summary_requires_only_the_four_actual_ranks(self):
        with tempfile.TemporaryDirectory() as output:
            root = Path(output) / "gpu_peak"
            root.mkdir()
            for rank in range(4):
                (root / f"actor_rank_{rank}.json").write_text(
                    json.dumps(
                        {"status": "complete", "update_peak_allocated_gib": 18.0}
                    )
                )
            report = write_peak_summary(Path(output), world_size=4)
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["missing_actor_ranks"], [])
            self.assertEqual(len(report["actor_reports"]), 4)

    def setUp(self):
        quiet = patch("spatial_wovr_gpu_peak.logger")
        quiet.start()
        self.addCleanup(quiet.stop)

    def test_backward_and_optimizer_peaks_exclude_cache_warmup(self):
        with tempfile.TemporaryDirectory() as output:
            instance, cuda = actor(output), FakeCuda()
            profile = ActorUpdatePeak(instance, SimpleNamespace(cuda=cuda))
            profile.start()
            for _ in range(8):
                cuda.peak_allocated = 4 * GIB
                profile.after_backward()
            instance.optimizer_steps = 1
            cuda.peak_allocated = 6 * GIB
            report = profile.finish(metrics={"actor/grad_norm": 0.5})
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["backward_peak_allocated_gib"], 4)
            self.assertEqual(report["update_peak_allocated_gib"], 6)
            self.assertEqual(report["update_peak_reserved_gib"], 8)
            self.assertEqual(report["backward_calls_completed"], 8)

    def test_oom_retains_a_failed_report_without_inventing_a_completed_update(self):
        with tempfile.TemporaryDirectory() as output:
            instance, cuda = actor(output), FakeCuda()
            profile = ActorUpdatePeak(instance, SimpleNamespace(cuda=cuda))
            profile.start()
            cuda.peak_allocated = 5 * GIB
            report = profile.finish(error=RuntimeError("CUDA out of memory"))
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["optimizer_step_counter_delta"], 0)
            self.assertEqual(report["update_peak_allocated_gib"], 5)
            self.assertIsNone(report["backward_peak_allocated_gib"])
            self.assertTrue((Path(output) / "gpu_peak/actor_rank_0.json").exists())

    def test_missing_backwards_or_more_than_one_update_is_rejected(self):
        for backwards, updates in [(0, 1), (8, 2)]:
            with tempfile.TemporaryDirectory() as output:
                instance, cuda = actor(output), FakeCuda()
                profile = ActorUpdatePeak(instance, SimpleNamespace(cuda=cuda))
                profile.start()
                for _ in range(backwards):
                    profile.after_backward()
                instance.optimizer_steps = updates
                with self.assertRaises(RuntimeError):
                    profile.finish(metrics={"actor/grad_norm": 1})

    def test_native_actor_runs_eight_backwards_then_measures_before_cache_refill(self):
        cuda = FakeCuda()

        class BaseActor:
            def run_training(self):
                for i in range(8):
                    self.train_micro_batch(i)
                self.optimizer_steps += 1
                cuda.peak_allocated = 6 * GIB
                return {"actor/grad_norm": 0.5}

            def train_micro_batch(self, batch):
                cuda.peak_allocated = 4 * GIB

        spec = importlib.util.spec_from_file_location(
            "profile_actor_test",
            Path(__file__).resolve().parents[1] / "spatial_grpo_actor.py",
        )
        module = importlib.util.module_from_spec(spec)
        with patch.dict(
            sys.modules,
            {
                "torch": SimpleNamespace(cuda=cuda),
                "rlinf.workers.actor.fsdp_actor_worker": SimpleNamespace(
                    EmbodiedFSDPActor=BaseActor
                ),
            },
        ):
            spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as output:
            instance = module.ReservedEmbodiedFSDPActor()
            instance.__dict__.update(actor(output).__dict__)
            instance._cuda_cache_reservation = SimpleNamespace(
                retain=lambda phase: setattr(cuda, "peak_allocated", 8 * GIB)
            )
            instance.run_training()
            report = json.loads(
                (Path(output) / "gpu_peak/actor_rank_0.json").read_text()
            )
            self.assertEqual(report["backward_calls_completed"], 8)
            self.assertEqual(report["update_peak_allocated_gib"], 6)
            self.assertEqual(cuda.peak_allocated, 8 * GIB)

    def test_incomplete_eight_rank_run_is_labeled_incomplete(self):
        with tempfile.TemporaryDirectory() as output:
            report = write_peak_summary(Path(output))
            self.assertEqual(report["status"], "incomplete_or_failed")
            self.assertIsNone(report["max_actor_update_allocated_gib"])
            self.assertEqual(report["missing_actor_ranks"], list(range(8)))

    def test_probe_config_is_one_native_global_batch_without_save_or_eval(self):
        cfg = OmegaConf.load(
            Path(__file__).resolve().parents[1] / "config/spatial_wovr_pi05_8gpu.yaml"
        )
        cfg.runner.max_epochs = cfg.runner.max_steps = 1
        cfg.runner.save_interval = cfg.runner.val_check_interval = -1
        cfg.env.train.rollout_epoch = 1
        cfg.env.train.max_steps_per_rollout_epoch = 128
        cfg.actor.gpu_memory_profile.enabled = True
        cfg.actor.gpu_memory_profile.expected_optimizer_updates = 1
        self.assertEqual(validate_batch(cfg), cfg.actor.global_batch_size)
        self.assertEqual(
            cfg.actor.global_batch_size // 8 // cfg.actor.micro_batch_size, 8
        )
        cfg.env.train.rollout_epoch = 8
        with self.assertRaisesRegex(ValueError, "One-update"):
            validate_batch(cfg)


if __name__ == "__main__":
    unittest.main()
