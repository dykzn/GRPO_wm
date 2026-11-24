"""Unit tests for Task- & Class-Balanced TaskEmbedResnetRewModel training and WanEnv scoring."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

EXPERIMENTS = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENTS.parents[1]
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(REPO_ROOT / "sources/diffsynth"))

from train_task_balanced_rm import (
    DEFAULT_BASELINE_RM,
    DEFAULT_MODEL_SOURCE,
    build_task_balanced_index,
    forward_task_embed_probs,
    load_reward_module,
    sample_balanced_epoch,
    set_trainable_layers,
    warm_start_task_embed_from_resnet,
)


def test_warm_start_and_layer_freezing_and_continuous_forward():
    rm_mod = load_reward_module(DEFAULT_MODEL_SOURCE)
    model = rm_mod.TaskEmbedResnetRewModel(
        checkpoint_path=None,
        num_tasks=10,
        task_embed_dim=64,
        task_suite_name="libero_spatial",
    )
    copied = warm_start_task_embed_from_resnet(model, DEFAULT_BASELINE_RM)
    assert copied >= 100

    set_trainable_layers(model, unfreeze_from_block=4)
    # Blocks 0, 1, 2 (b1..b3) frozen; blocks 3, 4 (b4, b5) + task_embedding + fusion_layer trainable.
    for idx in (0, 1, 2):
        assert all(not p.requires_grad for p in model.visual_encoder[idx].parameters())
    for idx in (3, 4):
        assert all(p.requires_grad for p in model.visual_encoder[idx].parameters())
    assert model.task_embedding.weight.requires_grad is True
    assert all(p.requires_grad for p in model.fusion_layer.parameters())

    model.eval()
    x = torch.randn(4, 3, 256, 256).clamp(-1.0, 1.0)
    t_ids = torch.tensor([0, 4, 7, 9], dtype=torch.long)
    probs = forward_task_embed_probs(model, x, t_ids)
    assert probs.shape == (4,)
    assert torch.all((probs >= 0.0) & (probs <= 1.0))
    # Continuous probabilities, not rounded 0/1.
    assert not torch.all((probs == 0.0) | (probs == 1.0))


def test_task_balanced_sampling_and_hard_negative_weighting(tmp_path):
    # Create synthetic metadata across all 10 tasks with wildly skewed success/failure counts.
    episodes = []
    ep_id = 0
    for t in range(10):
        # Task 0 has 10 succ / 1 fail; Task 4 has 2 succ / 4 fail.
        n_succ = 10 if t == 0 else 2
        n_fail = 1 if t == 0 else 4
        for _ in range(n_succ):
            episodes.append({"episode_index": ep_id, "task_id": t, "frames": 60, "success": True})
            probs = np.full(60, 0.1, dtype=np.float32)
            probs[-5:] = 0.95 if t != 4 else 0.70  # Task 4 missed by baseline
            np.save(tmp_path / f"episode_{ep_id:04d}.npy", probs)
            ep_id += 1
        for _ in range(n_fail):
            episodes.append({"episode_index": ep_id, "task_id": t, "frames": 60, "success": False})
            probs = np.full(60, 0.05, dtype=np.float32)
            probs[50] = 0.92  # hard false-positive frame
            np.save(tmp_path / f"episode_{ep_id:04d}.npy", probs)
            ep_id += 1

    pools = build_task_balanced_index(
        episodes,
        baseline_probs_dir=tmp_path,
        pos_tail_frames=5,
        pre_success_margin=20,
        hard_neg_fp_weight=5.0,
        hard_neg_late_weight=3.0,
    )
    # Hard FP frame (index 50) must have strictly higher sampling probability than early fail frame (index 5).
    _, fr_arr, w_arr = pools[0]["hard_neg"]
    w_fp = float(w_arr[fr_arr == 50][0])
    w_early = float(w_arr[fr_arr == 5][0])
    assert w_fp == pytest.approx(5.0 * w_early, rel=1e-5)

    rng = np.random.default_rng(0)
    records = sample_balanced_epoch(pools, samples_per_task=40, pos_frac=0.5, hard_neg_frac=0.3, rng=rng)
    assert len(records) == 400
    for t in range(10):
        task_recs = [r for r in records if r[2] == t]
        assert len(task_recs) == 40
        assert sum(1 for r in task_recs if r[3] == 1.0) == 20
        assert sum(1 for r in task_recs if r[3] == 0.0) == 20


def test_model_task_id_uses_instruction_mapping_not_raw_libero_id():
    """Verify training/scoring use instruction_to_task_id (WanEnv contract), not raw libero task_id."""
    from train_task_balanced_rm import score_episodes

    rm_mod = load_reward_module(DEFAULT_MODEL_SOURCE)
    # LIBERO task 1 ("next to the ramekin") maps to alphabetical model_task_id 5 in LIBERO_SPATIAL_TASKS.
    desc_t1 = "pick up the black bowl next to the ramekin and place it on the plate"
    mapped = int(rm_mod.instruction_to_task_id(desc_t1, "libero_spatial").item())
    assert mapped == 5

    # Build a tiny fake model whose output equals task_id / 10.0 so we observe which ID was passed.
    class _SpyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.visual_encoder = lambda x: torch.zeros((x.shape[0], 1), device=x.device)
            self.task_embedding = lambda tids: tids.float().unsqueeze(-1) / 10.0
            self.fusion_layer = lambda z: z[:, 1:2]

    ep = {"episode_index": 0, "task_id": 1, "model_task_id": mapped, "frames": 3, "success": True}
    pools = build_task_balanced_index([ep], baseline_probs_dir=None, pos_tail_frames=2)
    assert len(pools[5]["pos"][0]) == 2
    assert len(pools[1]["pos"][0]) == 0

    frames = [np.zeros((3, 16, 16, 3), dtype=np.uint8)]
    probs = score_episodes(_SpyModel(), [ep], frames, device=torch.device("cpu"))
    assert np.allclose(probs[0], 0.5)
