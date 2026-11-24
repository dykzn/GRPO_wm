"""Count usable GRPO data without changing rewards, masks, or gradients."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch

RATIO_FIELDS = {
    "actor/ratio": "actor/ratio_valid",
    "actor/ratio_abs": "actor/ratio_abs_valid",
    "actor/clipped_ratio": "actor/clipped_ratio_valid",
}


def sum_across_ranks(values: list[float], device: Any) -> list[float]:
    """Sum counts and weighted numerators once, including ranks with no data."""
    totals = torch.tensor(values, dtype=torch.float64, device=device)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)
    return totals.cpu().tolist()


@torch.no_grad()
def count_grpo_groups(
    rewards: torch.Tensor,
    prefix_mask: torch.Tensor,
    training_mask: torch.Tensor,
    group_size: int,
) -> list[float]:
    """Count groups before filtering, retained groups, and retained score diversity.

    Args:
        rewards: Native processed rewards shaped [time, trajectories, actions].
        prefix_mask: Native first-terminal mask before reward-group filtering.
        training_mask: Actual loss mask after all native filtering.
        group_size: Consecutive trajectories sharing one GRPO start state.

    Returns:
        Total, retained, and retained nonconstant-score group counts. Scores
        exclude the suffix after first success, just like native filtering.
    """
    if rewards.ndim != 3 or group_size < 1 or rewards.shape[1] % group_size:
        raise ValueError("GRPO diagnostics require complete trajectory groups")
    prefix_mask = torch.broadcast_to(prefix_mask.bool(), rewards.shape)
    training_mask = torch.broadcast_to(training_mask.bool(), rewards.shape)
    scores = rewards.masked_fill(~prefix_mask, 0).sum(dim=(0, 2))
    grouped = scores.reshape(-1, group_size)
    kept = training_mask.any(dim=0).any(dim=-1).reshape(-1, group_size).any(dim=1)
    diverse = grouped.amax(dim=1) - grouped.amin(dim=1) > 1e-6
    return [float(len(grouped)), float(kept.sum()), float((kept & diverse).sum())]


def group_metrics(counts: list[float]) -> dict[str, float]:
    """Derive ratios only after group counts have been summed across ranks."""
    total, kept, signal = counts
    return {
        "grpo_groups_total": total,
        "grpo_groups_kept": kept,
        "grpo_groups_signal": signal,
        "grpo_group_keep_frac": kept / total if total else float("nan"),
        "grpo_group_signal_frac": signal / total if total else float("nan"),
    }


class MicroBatchDiagnostics:
    """Accumulate native metric means weighted by valid action count.

    The Spatial policy uses seven log-probability dimensions per action on
    every rank, so action weighting also gives the valid-token weighted mean.
    Empty micro-batches still execute native training; only reporting changes.
    """

    def __init__(self) -> None:
        self.micro_count = 0
        self.empty_count = 0
        self.action_count = 0
        self.valid_count = 0
        self.numerators = {key: 0.0 for key in RATIO_FIELDS}

    def record(self, mask: torch.Tensor, metrics: dict[str, float]) -> None:
        """Observe the just-completed native micro-batch without keeping tensors."""
        valid = int(mask.detach().count_nonzero())
        self.micro_count += 1
        self.action_count += mask.numel()
        self.valid_count += valid
        if not valid:
            self.empty_count += 1
            return
        for key in RATIO_FIELDS:
            self.numerators[key] += float(metrics[key]) * valid

    def counts(self) -> list[float]:
        """Return additive statistics, never per-rank averages."""
        return [
            self.micro_count,
            self.empty_count,
            self.action_count,
            self.valid_count,
            *self.numerators.values(),
        ]

    @staticmethod
    def summarize(counts: list[float]) -> dict[str, float]:
        """Report undefined valid-only means as NaN when every mask is empty."""
        micro, empty, actions, valid, *numerators = counts
        result = {
            "actor/micro_count": micro,
            "actor/empty_micro_count": empty,
            "actor/empty_micro_frac": empty / micro if micro else float("nan"),
            "actor/valid_actions": valid,
            "actor/valid_action_frac": valid / actions if actions else float("nan"),
        }
        for name, numerator in zip(RATIO_FIELDS.values(), numerators, strict=True):
            result[name] = numerator / valid if valid else float("nan")
        return result

    def reduce(self, device: Any) -> dict[str, float]:
        """Collect all ranks, including those with no valid training actions."""
        return self.summarize(sum_across_ranks(self.counts(), device))


def successful_frame_metrics(shards: list[dict]) -> dict[str, float]:
    """Exclude failed episodes' -1 sentinel from the first-success frame mean."""
    frame_sum, count, present = 0.0, 0, False
    for shard in shards:
        if not shard or "first_success_step" not in shard:
            continue
        present = True
        values = shard["first_success_step"]
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        values = np.asarray(values).reshape(-1)
        successful = values[np.isfinite(values) & (values >= 1)]
        frame_sum += float(successful.sum())
        count += len(successful)
    if not present:
        return {}
    return {
        "first_success_mean": frame_sum / count if count else float("nan"),
        "first_success_count": float(count),
    }


def bind_environment_diagnostics(runner_module: Any) -> None:
    """Extend only this launch's runner aggregation, leaving native utilities intact."""
    original: Callable = runner_module.compute_evaluate_metrics
    if getattr(original, "_spatial_diagnostics", False):
        return

    def aggregate(shards: list[dict]) -> dict:
        metrics = original(shards)
        metrics.update(successful_frame_metrics(shards))
        return metrics

    aggregate._spatial_diagnostics = True
    runner_module.compute_evaluate_metrics = aggregate
