"""Validate real-evaluation coverage, checkpoint ordering and schedule."""

import csv
import json
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock

import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spatial_wovr_eval import SpatialEvaluationMixin, summarize_real_evaluation
from spatial_wovr_launch import validate_batch


def shards(world_size=8):
    rows = [(task, trial, trial % 2 == 0) for trial in range(50) for task in range(10)]
    result = []
    for rank in range(world_size):
        part = np.array(rows[rank::world_size])
        result.append(
            {"task_id": part[:, 0], "trial_id": part[:, 1], "success_once": part[:, 2]}
        )
    return result


class FakeRunner(SpatialEvaluationMixin):
    def __init__(self):
        self.cfg = OmegaConf.create(
            {"runner": {"val_check_interval": 5, "save_interval": 5}}
        )
        self.max_steps = 50
        self.events = []
        self.timer = lambda name: nullcontext()
        self.metric_logger = Mock()

    def _save_checkpoint(self):
        self.events.append((self.global_step, "save"))

    def update_rollout_weights(self):
        self.events.append((self.global_step, "sync"))

    def evaluate(self):
        self.events.append((self.global_step, "eval"))
        return {"success_once": 0.5, "num_trajectories": 500}


class RealEvaluationTests(unittest.TestCase):
    def test_four_rank_shards_still_cover_10_tasks_and_50_unique_trials_each(self):
        report = summarize_real_evaluation(shards(world_size=4))
        self.assertEqual(report["episodes"], 500)
        self.assertEqual(report["success"], 250)
        self.assertTrue(
            all(value["episodes"] == 50 for value in report["per_task"].values())
        )

    def test_real_evaluate_exports_500_results_for_the_saved_step(self):
        import torch

        class Runner(SpatialEvaluationMixin):
            pass

        runner = Runner()
        runner.global_step = 5
        runner.env_channel, runner.rollout_channel = object(), object()
        runner.env, runner.rollout, runner.logger = Mock(), Mock(), Mock()
        data = [
            {key: torch.as_tensor(value) for key, value in shard.items()}
            for shard in shards()
        ]
        runner.env.evaluate.return_value.wait.return_value = data
        with tempfile.TemporaryDirectory() as tmp:
            runner.cfg = OmegaConf.create(
                {
                    "runner": {"logger": {"log_path": tmp, "experiment_name": "test"}},
                    "env": {"eval": {"max_episode_steps": 240}},
                }
            )
            metrics = runner.evaluate()
            directory = Path(tmp) / "test/evaluations/global_step_5"
            report = json.loads((directory / "metrics.json").read_text())
            self.assertEqual(report["global_step"], 5)
            self.assertTrue(report["checkpoint"].endswith("checkpoints/global_step_5"))
            self.assertEqual(report["episodes"], 500)
            self.assertEqual(metrics["num_trajectories"], 500)
            with (directory / "episodes.csv").open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 500)

    def test_uneven_eight_rank_shards_preserve_exactly_500_episodes(self):
        report = summarize_real_evaluation(shards())
        self.assertEqual(report["episodes"], 500)
        self.assertEqual(report["success"], 250)
        self.assertEqual(report["success_rate"], 0.5)
        self.assertTrue(
            all(value["episodes"] == 50 for value in report["per_task"].values())
        )

    def test_missing_duplicate_out_of_range_or_nonbinary_data_is_rejected(self):
        cases = []
        data = shards()
        data[0] = {key: value[:-1] for key, value in data[0].items()}
        cases.append(data)
        data = shards()
        data.append(data[0])
        cases.append(data)
        for key, bad in [("task_id", 10), ("trial_id", 50), ("success_once", 2)]:
            data = shards()
            data[0][key][0] = bad
            cases.append(data)
        data = shards()
        del data[0]["trial_id"]
        cases.append(data)
        for data in cases:
            with self.assertRaises(RuntimeError):
                summarize_real_evaluation(data)

    def test_checkpoints_and_real_evaluations_run_at_steps_5_through_50(self):
        runner = FakeRunner()
        for step in range(1, 51):
            runner.global_step = step
            runner._maybe_eval_and_checkpoint(step - 1)
        self.assertEqual(
            runner.events,
            [
                (step, event)
                for step in range(5, 51, 5)
                for event in ("save", "sync", "eval")
            ],
        )
        self.assertEqual(
            [call.kwargs["step"] for call in runner.metric_logger.log.call_args_list],
            list(range(5, 51, 5)),
        )

    def test_checkpoint_is_saved_before_a_failed_evaluation(self):
        runner = FakeRunner()
        runner.global_step = 5
        runner.evaluate = Mock(side_effect=RuntimeError("incomplete real eval"))
        with self.assertRaises(RuntimeError):
            runner._maybe_eval_and_checkpoint(4)
        self.assertEqual(runner.events[0], (5, "save"))

    def test_one_cycle_smoke_can_disable_real_evaluation(self):
        runner = FakeRunner()
        runner.max_steps = runner.global_step = 1
        runner.cfg.runner.val_check_interval = -1
        runner.cfg.runner.save_interval = 1
        runner._maybe_eval_and_checkpoint(0)
        self.assertEqual(runner.events, [(1, "save")])

    def test_evaluation_budget_must_cover_all_trials_even_if_every_trial_times_out(
        self,
    ):
        cfg = OmegaConf.load(
            Path(__file__).resolve().parents[1] / "config/spatial_wovr_pi05_8gpu.yaml"
        )
        validate_batch(cfg)
        cfg.env.eval.rollout_epoch = 1
        with self.assertRaisesRegex(ValueError, "8 waves"):
            validate_batch(cfg)

    def test_fake_results_cannot_substitute_for_simulator_evaluation(self):
        cfg = OmegaConf.load(
            Path(__file__).resolve().parents[1] / "config/spatial_wovr_pi05_8gpu.yaml"
        )
        for key, value in [
            ("env_type", "wan_wm"),
            ("task_suite_name", "libero_10"),
            ("episodes_per_task", 49),
            ("ignore_terminations", True),
            ("task_id_filter", [0]),
        ]:
            altered = OmegaConf.create(OmegaConf.to_container(cfg))
            OmegaConf.update(altered.env.eval, key, value)
            with self.assertRaises(ValueError):
                validate_batch(altered)


if __name__ == "__main__":
    unittest.main()
