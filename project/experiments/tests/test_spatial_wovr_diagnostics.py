"""Check weighted logging across empty batches, unequal ranks, and failures."""

import importlib.util
import math
import multiprocessing
import os
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spatial_wovr_diagnostics import (
    RATIO_FIELDS,
    MicroBatchDiagnostics,
    bind_environment_diagnostics,
    count_grpo_groups,
    group_metrics,
    successful_frame_metrics,
    sum_across_ranks,
)


def ratio_metrics(ratio):
    return {
        "actor/ratio": ratio,
        "actor/ratio_abs": abs(ratio - 1),
        "actor/clipped_ratio": max(0.8, min(1.28, ratio)),
    }


def distributed_worker(rank, rendezvous, queue):
    """Use real Gloo collectives with different valid counts on each rank."""
    torch.set_num_threads(1)
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    torch.distributed.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2
    )
    try:
        diagnostics = MicroBatchDiagnostics()
        if rank == 0:
            diagnostics.record(torch.tensor([[True, False, False]]), ratio_metrics(1.1))
            diagnostics.record(torch.zeros(1, 3, dtype=torch.bool), {})
        else:
            diagnostics.record(torch.ones(1, 3, dtype=torch.bool), ratio_metrics(0.9))
        weighted = diagnostics.reduce("cpu")
        diagnostics = MicroBatchDiagnostics()
        if rank == 0:
            diagnostics.record(torch.zeros(1, 2, dtype=torch.bool), {})
        else:
            diagnostics.record(torch.ones(1, 2, dtype=torch.bool), ratio_metrics(1.0))
        empty_rank = diagnostics.reduce("cpu")
        groups = group_metrics(
            sum_across_ranks([2, 0, 0] if rank == 0 else [3, 2, 1], "cpu")
        )
        queue.put((rank, weighted, empty_rank, groups))
    finally:
        torch.distributed.destroy_process_group()


class DiagnosticsTests(unittest.TestCase):
    def test_group_scores_ignore_terminal_suffix_and_distinguish_kept_from_signal(self):
        # Two varying groups, one all-failure group, one all-success group.
        scores = torch.tensor([0, 5, 0, 5, 0, 0, 0, 0, 5, 5, 5, 5, 0, 0, 5, 5.0])
        rewards = torch.stack([scores, torch.full_like(scores, 999)]).unsqueeze(-1)
        prefix = torch.ones_like(rewards, dtype=torch.bool)
        prefix[1] = False
        training = prefix.clone()
        training[:, 4:8] = False
        training[:, 8:12] = False
        originals = [value.clone() for value in [rewards, prefix, training]]
        self.assertEqual(count_grpo_groups(rewards, prefix, training, 4), [4, 2, 2])
        for value, original in zip([rewards, prefix, training], originals, strict=True):
            torch.testing.assert_close(value, original)
        # If native filtering retains a constant group, do not count it as signal.
        training[:, 8:12] = prefix[:, 8:12]
        self.assertEqual(count_grpo_groups(rewards, prefix, training, 4), [4, 3, 2])
        torch.testing.assert_close(rewards, originals[0])
        torch.testing.assert_close(prefix, originals[1])
        self.assertFalse(originals[2][:, 8:12].any())

    def test_partial_groups_are_rejected(self):
        values = torch.zeros(2, 7, 8)
        with self.assertRaisesRegex(ValueError, "complete trajectory groups"):
            count_grpo_groups(values, values.bool(), values.bool(), 8)

    @unittest.skipUnless(
        importlib.util.find_spec("rlinf") is not None,
        "Native integration requires the launcher's runtime PYTHONPATH",
    )
    def test_native_reward_filter_outputs_are_unchanged_by_group_diagnostics(self):
        from omegaconf import OmegaConf
        from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
        from spatial_wovr_actor import WorldModelFSDPActor

        native = object.__new__(EmbodiedFSDPActor)
        diagnostic = object.__new__(WorldModelFSDPActor)
        cfg = OmegaConf.create(
            {
                "env": {
                    "train": {
                        "rollout_epoch": 1,
                        "auto_reset": False,
                        "ignore_terminations": False,
                    }
                },
                "algorithm": {
                    "reward_type": "action_level",
                    "filter_rewards": True,
                    "group_size": 8,
                    "rewards_lower_bound": 0.5,
                    "rewards_upper_bound": 4.5,
                },
            }
        )
        native.cfg = diagnostic.cfg = cfg
        rewards = torch.zeros(2, 24, 8)
        # Group 0: all failure; group 1: mixed; group 2: all success.
        rewards[0, 8, 0] = 5
        rewards[0, 16:, 0] = 5
        dones = torch.zeros(3, 24, 8, dtype=torch.bool)
        dones[1, 8, :] = True
        dones[1, 16:, :] = True
        batch = {"rewards": rewards, "dones": dones}
        original = {key: value.clone() for key, value in batch.items()}
        expected = native._process_received_rollout_batch(batch)
        actual = diagnostic._process_received_rollout_batch(batch)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key])
        for key, value in batch.items():
            torch.testing.assert_close(value, original[key])
        self.assertEqual(diagnostic._grpo_group_counts, [3, 1, 1])

    def test_empty_micro_batches_do_not_dilute_ratio(self):
        diagnostics = MicroBatchDiagnostics()
        for _ in range(3):
            diagnostics.record(torch.ones(2, 8, dtype=torch.bool), ratio_metrics(1))
        diagnostics.record(torch.zeros(2, 8, dtype=torch.bool), {})
        result = diagnostics.reduce("cpu")
        self.assertEqual(result["actor/micro_count"], 4)
        self.assertEqual(result["actor/empty_micro_count"], 1)
        self.assertEqual(result["actor/empty_micro_frac"], 0.25)
        self.assertEqual(result["actor/valid_action_frac"], 0.75)
        self.assertEqual(result["actor/ratio_valid"], 1)
        self.assertEqual(result["actor/ratio_abs_valid"], 0)

    def test_all_empty_reports_missing_valid_means_instead_of_a_zero_ratio(self):
        diagnostics = MicroBatchDiagnostics()
        diagnostics.record(torch.zeros(2, 8, dtype=torch.bool), {})
        result = diagnostics.reduce("cpu")
        self.assertEqual(result["actor/empty_micro_frac"], 1)
        self.assertTrue(math.isnan(result["actor/ratio_valid"]))

    def test_real_distributed_sums_handle_unequal_weights_and_a_fully_empty_rank(self):
        try:
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
        except OSError as error:
            self.skipTest(f"Local distributed sockets unavailable: {error}")
        context = multiprocessing.get_context("spawn")
        with tempfile.TemporaryDirectory() as tmp:
            queue = context.Queue()
            workers = [
                context.Process(
                    target=distributed_worker,
                    args=(rank, str(Path(tmp) / "rendezvous"), queue),
                )
                for rank in range(2)
            ]
            try:
                for worker in workers:
                    worker.start()
                results = [queue.get(timeout=45) for _ in workers]
                for worker in workers:
                    worker.join(timeout=10)
                    self.assertEqual(worker.exitcode, 0)
            finally:
                for worker in workers:
                    if worker.is_alive():
                        worker.terminate()
                        worker.join(timeout=5)
                queue.close()
            for _, weighted, empty_rank, groups in results:
                self.assertAlmostEqual(weighted["actor/ratio_valid"], 0.95)
                self.assertAlmostEqual(weighted["actor/empty_micro_frac"], 1 / 3)
                self.assertEqual(weighted["actor/valid_actions"], 4)
                self.assertEqual(empty_rank["actor/ratio_valid"], 1)
                self.assertEqual(empty_rank["actor/empty_micro_frac"], 0.5)
                self.assertEqual(groups["grpo_groups_total"], 5)
                self.assertEqual(groups["grpo_groups_kept"], 2)
                self.assertEqual(groups["grpo_groups_signal"], 1)
                self.assertEqual(groups["grpo_group_signal_frac"], 0.2)

    def test_unequal_rank_statistics_are_summed_before_dividing(self):
        left, right = MicroBatchDiagnostics(), MicroBatchDiagnostics()
        left.record(torch.tensor([[True, False, False]]), ratio_metrics(1.1))
        left.record(torch.zeros(1, 3, dtype=torch.bool), {})
        right.record(torch.ones(1, 3, dtype=torch.bool), ratio_metrics(0.9))
        totals = [a + b for a, b in zip(left.counts(), right.counts(), strict=True)]
        result = MicroBatchDiagnostics.summarize(totals)
        self.assertAlmostEqual(result["actor/ratio_valid"], 0.95)
        self.assertAlmostEqual(result["actor/empty_micro_frac"], 1 / 3)
        self.assertEqual(result["actor/valid_actions"], 4)
        self.assertAlmostEqual(result["actor/valid_action_frac"], 4 / 9)
        empty, valid = MicroBatchDiagnostics(), MicroBatchDiagnostics()
        empty.record(torch.zeros(1, 2, dtype=torch.bool), {})
        valid.record(torch.ones(1, 2, dtype=torch.bool), ratio_metrics(1))
        totals = [a + b for a, b in zip(empty.counts(), valid.counts(), strict=True)]
        self.assertEqual(
            MicroBatchDiagnostics.summarize(totals)["actor/ratio_valid"], 1
        )

    def test_first_success_mean_weights_episodes_not_rank_means(self):
        shards = [
            {"first_success_step": torch.tensor([10, -1, -1])},
            {"first_success_step": np.array([20, 30, 40, -1])},
        ]
        result = successful_frame_metrics(shards)
        self.assertEqual(result["first_success_mean"], 25)
        self.assertEqual(result["first_success_count"], 4)
        self.assertEqual(shards[0]["first_success_step"].tolist(), [10, -1, -1])

    def test_missing_or_unsuccessful_episodes_are_distinguished(self):
        self.assertEqual(successful_frame_metrics([{"success_once": [False]}]), {})
        result = successful_frame_metrics([{"first_success_step": [-1, -1]}])
        self.assertEqual(result["first_success_count"], 0)
        self.assertTrue(math.isnan(result["first_success_mean"]))

    def test_runner_binding_keeps_existing_metrics_and_does_not_stack_wrappers(self):
        shards = [{"first_success_step": [12, -1]}]
        native = lambda values: {"first_success_step": 5.5, "num_trajectories": 2}
        runner = SimpleNamespace(compute_evaluate_metrics=native)
        bind_environment_diagnostics(runner)
        bound = runner.compute_evaluate_metrics
        bind_environment_diagnostics(runner)
        self.assertIs(runner.compute_evaluate_metrics, bound)
        result = bound(shards)
        self.assertEqual(result["first_success_mean"], 12)
        self.assertEqual(result["first_success_step"], 5.5)
        self.assertEqual(result["num_trajectories"], 2)
        self.assertNotIn("first_success_mean", native(shards))

    def test_actor_hooks_preserve_updates_and_reset_counters_each_outer_step(self):
        class BaseActor:
            def run_training(self):
                self.weight.grad = None
                metrics = {}
                for mask in [
                    torch.ones(2, 8, dtype=torch.bool),
                    torch.zeros(2, 8, dtype=torch.bool),
                ]:
                    self.train_micro_batch({"loss_mask": mask}, metrics, is_last=False)
                with torch.no_grad():
                    self.weight -= self.weight.grad * 0.1
                return {
                    key: sum(values) / len(values) for key, values in metrics.items()
                }

            def train_micro_batch(self, micro_batch, metrics, *, is_last):
                mask = micro_batch["loss_mask"]
                loss = (self.weight.square().expand_as(mask) * mask).mean()
                loss.backward()
                values = (
                    ratio_metrics(1) if mask.any() else dict.fromkeys(RATIO_FIELDS, 0)
                )
                for key, value in values.items():
                    metrics.setdefault(key, []).append(value)

        spec = importlib.util.spec_from_file_location(
            "diagnostic_actor_test",
            Path(__file__).resolve().parents[1] / "spatial_wovr_actor.py",
        )
        module = importlib.util.module_from_spec(spec)
        with patch.dict(
            sys.modules,
            {
                "spatial_grpo_actor": SimpleNamespace(
                    ReservedEmbodiedFSDPActor=BaseActor
                )
            },
        ):
            spec.loader.exec_module(module)
        actual, baseline = module.WorldModelFSDPActor(), BaseActor()
        for instance in [actual, baseline]:
            instance.device = "cpu"
            instance.weight = torch.nn.Parameter(torch.tensor(2.0))
            instance._cuda_cache_reservation = None
        for _ in range(2):
            expected = baseline.run_training()
            result = actual.run_training()
            torch.testing.assert_close(actual.weight, baseline.weight)
            torch.testing.assert_close(actual.weight.grad, baseline.weight.grad)
            for key in RATIO_FIELDS:
                self.assertEqual(result[key], expected[key])
            self.assertEqual(result["actor/ratio"], 0.5)
            self.assertEqual(result["actor/ratio_valid"], 1)
            self.assertEqual(result["actor/micro_count"], 2)
            self.assertEqual(result["actor/empty_micro_count"], 1)
            self.assertIsNone(actual._active_training_diagnostics)


if __name__ == "__main__":
    unittest.main()
