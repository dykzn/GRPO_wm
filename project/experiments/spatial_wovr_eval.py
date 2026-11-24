"""Require complete, repeatable real LIBERO evaluations at saved GRPO steps."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from kir_frame_pool import atomic_json


def summarize_real_evaluation(shards: list[dict]) -> dict:
    """Validate exactly the 10 Spatial tasks and 50 distinct trials per task."""
    episodes = {}
    for shard in shards:
        if not shard:
            continue
        required = {"task_id", "trial_id", "success_once"}
        if not required <= shard.keys():
            raise RuntimeError(
                f"Real evaluation lacks episode metadata: {required - shard.keys()}"
            )
        arrays = [
            np.asarray(shard[key]).reshape(-1)
            for key in ("task_id", "trial_id", "success_once")
        ]
        if len({len(value) for value in arrays}) != 1:
            raise RuntimeError("Real evaluation episode metadata lengths disagree")
        for task, trial, success in zip(*arrays, strict=True):
            if not np.isfinite([task, trial, success]).all():
                raise RuntimeError("Nonfinite real evaluation metadata")
            if task != int(task) or trial != int(trial) or success not in (0, 1):
                raise RuntimeError("Invalid real evaluation task/trial/success value")
            key = (int(task), int(trial))
            if key in episodes:
                raise RuntimeError(f"Duplicate real evaluation episode: {key}")
            episodes[key] = bool(success)
    expected = {(task, trial) for task in range(10) for trial in range(50)}
    if episodes.keys() != expected:
        missing = expected - episodes.keys()
        extra = episodes.keys() - expected
        raise RuntimeError(
            f"Real evaluation must complete exactly 10x50 trials; got {len(episodes)}, "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    rows = [
        {"task_id": task, "trial_id": trial, "success": success}
        for (task, trial), success in sorted(episodes.items())
    ]
    successes = sum(episodes.values())
    return {
        "source": "real LIBERO simulator success, not the world-model reward classifier",
        "episodes": 500,
        "success": successes,
        "failure": 500 - successes,
        "success_rate": successes / 500,
        "per_task": {
            str(task): {
                "episodes": 50,
                "success": sum(episodes[(task, trial)] for trial in range(50)),
                "success_rate": sum(episodes[(task, trial)] for trial in range(50))
                / 50,
            }
            for task in range(10)
        },
        "episode_results": rows,
    }


class SpatialEvaluationMixin:
    """Save before evaluating and retain auditable results for each checkpoint."""

    def evaluate(self):
        from rlinf.utils.metric_utils import compute_evaluate_metrics

        env_handle = self.env.evaluate(
            input_channel=self.env_channel, rollout_channel=self.rollout_channel
        )
        rollout_handle = self.rollout.evaluate(
            input_channel=self.rollout_channel, output_channel=self.env_channel
        )
        shards = [result for result in env_handle.wait() if result is not None]
        rollout_handle.wait()
        report = summarize_real_evaluation(shards)
        root = (
            Path(self.cfg.runner.logger.log_path)
            / self.cfg.runner.logger.experiment_name
        )
        report.update(
            global_step=int(self.global_step),
            checkpoint=str(root / f"checkpoints/global_step_{self.global_step}"),
            max_episode_steps=int(self.cfg.env.eval.max_episode_steps),
        )
        directory = root / "evaluations" / f"global_step_{self.global_step}"
        directory.mkdir(parents=True, exist_ok=True)
        atomic_json(directory / "metrics.json", report)
        with (directory / "episodes.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=["task_id", "trial_id", "success"]
            )
            writer.writeheader()
            writer.writerows(report["episode_results"])
        self.logger.info(
            f"Real Spatial evaluation step {self.global_step}: "
            f"{report['success']}/500 = {report['success_rate']:.2%}; {directory}"
        )
        return compute_evaluate_metrics(shards)

    def _maybe_eval_and_checkpoint(self, step: int) -> dict:
        from rlinf.utils.runner_utils import check_progress

        run_val, save_model, _ = check_progress(
            self.global_step,
            self.max_steps,
            self.cfg.runner.val_check_interval,
            self.cfg.runner.save_interval,
            1.0,
            run_time_exceeded=False,
        )
        # A failed simulation evaluation must not discard the trained policy.
        if save_model:
            with self.timer("save_checkpoint"):
                self._save_checkpoint()
        if not run_val:
            return {}
        with self.timer("eval"):
            self.update_rollout_weights()
            metrics = {f"eval/{key}": value for key, value in self.evaluate().items()}
            self.metric_logger.log(data=metrics, step=self.global_step)
        return metrics
