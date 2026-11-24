# WorldSample Route-A (`GRPO_wm`)

Dual-Stream Hybrid RL (**Real-Env GAE-PPO + Norm-Matched Conflict-Free WM-GRPO**) and **Task-Balanced Primary-View Reward Model** (`TaskEmbedResnetRewModel`) for low real-budget VLA post-training on LIBERO-Spatial.

## Core Framework

1. **Anchor-Amplifier Dual-Stream RL** ([spatial_wovr_hybrid.py](project/experiments/spatial_wovr_hybrid.py), [spatial_wovr_pi05_8gpu_hybrid_ppo_grpo.yaml](project/experiments/config/spatial_wovr_pi05_8gpu_hybrid_ppo_grpo.yaml))
   - **Real Stream (GAE-PPO)**: Low-budget unbiased real environment rollouts (e.g., 32 eps/step × 16 steps = 512 real episodes from Step 0 SFT); exclusively updates the independent `ValueHead` (`compute_values=True`).
   - **WM Stream (Group-Relative GRPO)**: High-throughput closed-loop world-model rollouts (Wan 2.2 primary + Cosmos wrist, 128 envs × 4 epochs = 512 WM eps/step, `compute_values=False`) with `First-Success Mask` and signal-group filtering.
   - **Norm-Matched Conflict-Free Gradient Fusion**: Cross-rank FSDP global L2 norm matching + `cosine_gate` (hard fuse when `cos < 0`) or `pcgrad` (orthogonal projection), using CPU-stashed real gradients for zero extra GPU VRAM.
   - **Pacing**: `critic_warmup_steps=8` (Steps 1–2 skip WM) followed by deterministic `linear_ramp` (`ramp_steps=4`, `0 -> rho_max`).

2. **Task- & Class-Balanced Primary-View Reward Model** ([train_task_balanced_rm.py](project/experiments/train_task_balanced_rm.py), [world_model_wan_env.py](project/third_party/RLinf/rlinf/envs/world_model/world_model_wan_env.py))
   - Warm-started from `resnet_rm.pth` into `TaskEmbedResnetRewModel` (primary camera only, no wrist view).
   - Strict `instruction_to_task_id(..., "libero_spatial")` alignment, per-task & class-balanced hard-negative mining, and continuous probability scoring in `WanEnv`.
   - Held-out 100-episode validation (`@0.90`): cuts FPR in half (`14.3% -> 7.1%`, regularized `b5`-only variant `0.0%`) and raises Task 4 recall from `79.0%` to `92.1%` (`94.7%` in `v2`).

## Key Results (LIBERO-Spatial, Step 0 SFT, 500-Trial Real Eval)

| Step (Real Episodes) | Baseline Real PPO (`rho=0.0`) | Route-A `cosine_gate` + TB-RM v1 (`rho=0.30`) | Gain |
| :--- | :---: | :---: | :---: |
| Step 4 (128 real eps) | `82.8%` | **`83.2%`** | `+0.4%` |
| Step 8 (256 real eps) | `82.4%` | **`83.8%`** | `+1.4%` |
| Step 12 (384 real eps) | `83.0%` | **`85.0%`** | `+2.0%` |
| Step 16 (512 real eps) | `84.2%` | *(PCGrad 16-step run in progress)* | — |

## Documentation & Reports

- Concise 5-page Core Framework overview: [docs/WorldSample_Core_Framework.pdf](docs/WorldSample_Core_Framework.pdf)
- Full Technical Blueprint: [docs/WorldSample_RouteA_Technical_Blueprint.pdf](docs/WorldSample_RouteA_Technical_Blueprint.pdf)
- Task-Balanced RM evaluation reports: [results/task_balanced_rm_v1/](results/task_balanced_rm_v1/) and [results/task_balanced_rm_v2_b5/](results/task_balanced_rm_v2_b5/)

## Quick Usage (on the Reproduction Bundle)

```bash
# Baseline (512 real eps PPO only)
SPATIAL_WOVR_TMP_ROOT=/data/dyk/tmp \
SPATIAL_WOVR_HYBRID_PPO_GRPO=1 \
SPATIAL_WOVR_HYBRID_RHO_MAX=0.0 \
SPATIAL_WOVR_EXTRA_OVERRIDES="runner.max_steps=16 runner.max_epochs=16 runner.val_check_interval=4 runner.save_interval=4" \
python3 -u reproduction/portable.py train --steps 16 --output /data/dyk/worldsample_runs/hybrid_base

# Route-A Hybrid (cosine_gate or pcgrad + Task-Balanced RM v1)
SPATIAL_WOVR_TMP_ROOT=/data/dyk/tmp \
SPATIAL_WOVR_HYBRID_PPO_GRPO=1 \
SPATIAL_WOVR_HYBRID_RHO_MAX=0.30 \
SPATIAL_WOVR_HYBRID_CONFLICT_MODE=pcgrad \
SPATIAL_WOVR_EXTRA_OVERRIDES="env.train.reward_model.type=TaskEmbedResnetRewModel env.train.reward_model.from_pretrained=/path/to/task_balanced_resnet_rm.pth env.train.success_reward_threshold=0.9 runner.max_steps=16 runner.max_epochs=16 runner.val_check_interval=4 runner.save_interval=4" \
python3 -u reproduction/portable.py train --steps 16 --output /data/dyk/worldsample_runs/hybrid_exp
```
