# Cosmos LIBERO Native Contract Audit

**Date**: 2026-07-29
**Scope**: C001–C003 / M0
**Result**: M0 audit complete; native `1 reference + 4 memory -> 8 future frames` gate is blocked.

## Outcome

The released `nvidia/Cosmos-Policy-LIBERO-Predict2-2B` checkpoint is a valid action-conditioned endpoint world model, but it is not a drop-in WoVR-style clip generator:

- input: one current primary image, one current wrist image, current 9-D proprio, language, and a 16×7 action chunk;
- output: one primary image, one wrist image, and one 9-D proprio at `t+16`, plus one scalar discounted-return estimate;
- no native 1+4 temporal observation history;
- no native 8-frame temporal future clip.

The model's nine latent time slots are semantic modality slots:

```text
blank, current proprio, current wrist, current primary, action,
future proprio, future wrist, future primary, value
```

They must not be decoded or relabeled as nine consecutive future times. The tokenizer's 33 raw image positions are four duplicates per semantic slot, not a temporal future video.

## C001: checkpoint and temporal layout

- Checkpoint: `${COSMOS_CHECKPOINT_ROOT}/Cosmos-Policy-LIBERO-Predict2-2B.pt`
- Size: 3,913,017,345 bytes
- Config: `cosmos_predict2_2b_480p_libero__inference_only`
- Action horizon: 16 low-level actions
- Dataset rate: 20 Hz
- Endpoint: `t+16`, or 0.8 seconds after the current observation
- Native future images: one frame per camera

Formal runs must use a Pi05 horizon of 16. The default RLinf Cosmos config and Pi05 actor config now both use 16 and fail if a short action chunk would require padding. Repeating the last Pi05 action remains available only when a legacy diagnostic explicitly opts into it.

Batch Cosmos diffusion uses one seed for the entire batch, not independent per-sample seeds. Reproducibility therefore requires a fixed batch seed, sample ordering, and batch shape. The batched route also depends on the workspace extension `cosmos_utils.get_action_batch`; this patch must be migrated together with WorldPilot and RLinf.

## C002: LeRobot alignment

All four complete suites use 20 Hz timestamps with frame, state, and action rows aligned one-to-one. Episode 0 was checked in every suite; parquet row counts equal video frame counts and timestamps advance by 0.05 seconds.

| Suite | Episodes | Frames | Complete path |
|---|---:|---:|---|
| Spatial | 432 | 52,970 | `${LIBERO_DATA_ROOT}/libero_spatial_no_noops_1.0.0_lerobot` |
| Object | 454 | 66,984 | `${LIBERO_DATA_ROOT}/libero_object_no_noops_1.0.0_lerobot` |
| Goal | 428 | 52,042 | `${LIBERO_DATA_ROOT}/libero_goal_no_noops_1.0.0_lerobot` |
| LIBERO-10 | 379 | 101,469 | `${LIBERO_DATA_ROOT}/libero_10_no_noops_1.0.0_lerobot` |
| Total | 1,693 | 273,465 | — |

This also resolves the historical “432 episodes” count: 432 is the complete Spatial suite only, not all LIBERO data.

The nested Spatial copy under `LEROBOT_LIBERO_DATA/libero/` is incomplete: it has 13 parquet files and no metadata or videos. `RLinf/examples/embodiment/config/env/cosmos_libero.yaml` now points explicitly to the complete sibling export.

For a leakage-free WoVR window at decision index `t`:

```text
reference = t-4
memory    = [t-3, t-2, t-1, t]
actions   = [t, ..., t+15]
Cosmos target = t+16
```

Thus `4 <= t <= T-17`.

## C003: value semantics

The target is not an environment reward and not merely diffusion confidence. It is the Monte Carlo return from the future frame:

```text
target = discounted sparse terminal success return at t+16
gamma  = 0.99 per low-level 20 Hz step
```

Training stores it in `[-1, 1]`; public inference maps it to `[0, 1]`. Failed episodes map to zero after unscaling.

With default masks disabled, its precise architectural semantics are neither pure `V(s)` nor pure `Q(s,a)`. In autoregressive value mode it is conditioned on current observation, the 16-step action, and the generated future state. It is best described as a discounted future-state return estimator and can only be used as a stop-gradient terminal bootstrap after calibration.

The prior WorldPilot external-action endpoint directly read the value latent produced during future-state generation. It now follows the official autoregressive planning route and calls `get_value_prediction` as a separate denoising stage conditioned on the generated future state. CPU-only single/batch/error-path tests are in `WorldPilot/cosmos_bridge/test_cosmos_value_route.py`.

## Gate decision

S101 cannot truthfully implement native `1+4 -> 8` with this checkpoint. Eight autoregressive calls would represent eight consecutive 16-action endpoints, spanning 128 low-level actions, and are not equivalent to WoVR's eight-frame future.

Recommended formal route: fine-tune a Cosmos temporal action-conditioned adapter/checkpoint with true 1+4 inputs and eight future-frame targets. A smaller alternative is to redefine the method around one `t+16` endpoint and retrain/calibrate its reward model, but that is no longer a WoVR 1+4-to-8 reproduction.

Machine-readable evidence is in `cosmos_libero_temporal_contract.json`; deterministic checks are in `test_cosmos_libero_temporal_contract.py`.
