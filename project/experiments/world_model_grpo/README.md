# World-model simulation -> GRPO

This directory is the collaboration-facing entry point for the current
hybrid experiment:

```text
Pi05 action chunk
    -> Wan primary future simulation
    -> Cosmos-WoVR wrist future simulation
    -> WoVR success reward
    -> GRPO group-relative advantage and actor update
```

The low-level runtime remains
`experiments/run_wovr_grpo_hybrid_wrist.sh`. The wrapper in this directory
does not replace it; it validates the inputs, creates a run-local Hydra
override file, and writes `run_manifest.json` into the result directory.

## Canonical command

For the current four-task comparison (tasks `0,3,5,9`) resumed from
`global_step_20` and continued for 20 more steps:

```bash
python experiments/world_model_grpo/run.py \\
  --wan-ckpt /absolute/path/to/merged_rlinf_step2000 \\
  --cosmos-ckpt /absolute/path/to/cosmos/checkpoints/iter_000005000 \\
  --pi05-root /path/to/checkpoints/pi05/RLinf-Pi05-LIBERO-SFT \\
  --resume-dir /absolute/path/to/checkpoints/global_step_20 \\
  --additional-steps 20 \\
  --gpu 6 \\
  --cosmos-gpu 6
```

The wrapper refuses to start if the target step is not greater than the
resume step, if the Wan/Cosmos/Pi05 artifacts are incomplete, if the task
filter differs between train and eval, or if the GPU is already above the
configured memory threshold. Use `--dry-run` to inspect the resolved manifest
and command without launching anything.

For a fresh run, use a target global step instead of a resume directory:

```bash
python experiments/world_model_grpo/run.py \\
  --wan-ckpt /absolute/path/to/merged_wan \\
  --cosmos-ckpt /absolute/path/to/cosmos/checkpoints/iter_000005000 \\
  --max-steps 20 \\
  --gpu 6
```

The wrapper runs the coordinator in the foreground. If a detached job is
needed, put the whole command in `screen` or another job manager.

## What belongs where

| Responsibility | Canonical location |
| --- | --- |
| Corrected VLA rollout and HDF5 manifest | external experiment workspace |
| Wan WMEvo training | external experiment workspace |
| Cosmos wrist WMEvo training | `WorldPilot`/Cosmos training launcher and its checkpoint directory |
| Four-task GRPO contract | `experiments/pace/goal_grpo_wmevo4_task0359_overrides.txt` |
| Coordinator and resident Cosmos server | `experiments/run_wovr_grpo_hybrid_wrist.sh` |
| Wan + Cosmos hybrid environment | `RLinf/rlinf/envs/world_model/world_model_wan_env.py` |
| Cosmos request/response adapter | `third_party/RLinf/rlinf/envs/world_model/cosmos_world_env.py` and `third_party/WorldPilot/cosmos_bridge/` |
| Camera convention boundary | `third_party/RLinf/rlinf/envs/world_model/cosmos_camera_conventions.py` plus the RLinf/WorldPilot/cosmos-policy wrappers |
| Current LIBERO bootstrap/state bridge | `third_party/RLinf/rlinf/envs/world_model/libero_state_bridge.py` |
| GRPO policy update | `third_party/RLinf/rlinf/algorithms/`, `third_party/RLinf/rlinf/workers/actor/`, and the external OpenPI dependency |

The merge target is the current dual-view Wan + Cosmos-WoVR wrist + Pi05 GRPO
path. LIBERO-Plus is outside the target. The state bridge is not a third view;
the current hybrid environment uses it to bootstrap reset images, task
metadata, and state, so it remains a runtime dependency until a separate
bootstrap refactor removes it. The runtime uses `RLinf`, not
`RLinf-kir-wrist`; the latter contains a similar working copy and should not
be merged as a second implementation of the same files.

## Artifact contract

- Wan checkpoint: `model-00001.safetensors` and `Wan2.2_VAE.pth`.
- Cosmos wrist checkpoint: a distributed checkpoint directory containing
  `model/.metadata`.
- Pi05 checkpoint: `model.safetensors`,
  `model_state_dict/full_weights.pt`, or
  `actor/model_state_dict/full_weights.pt`.
- Cosmos runtime assets: `success_only/dataset_statistics.json` and
  `success_only/t5_embeddings.pkl` under `--cosmos-dataset-root`.
- GRPO resume: an RLinf checkpoint directory, normally named
  `global_step_<N>`.

The wrapper copies the resolved override file to the result directory. The
manifest records all artifact paths, target steps, ports, GPUs, and the git
revision/dirty state of `RLinf`, `WorldPilot`, and `cosmos-policy`.

On another machine, source `scripts/env.sh` or set the values in a private
`.env` file. In particular, set `LIBERO_ROOT`, `RLINF_ENV`, `COSMOS_PYTHON`,
`CHECKPOINT_ROOT`, and `HYBRID_COSMOS_DATASET_ROOT` instead of editing the
launcher.

The root-level `cosmos_camera_conventions.py` is shared by all three runtime
repositories. It must travel with the RLinf camera wrapper; otherwise a clean
RLinf checkout will fail before the dual-view environment starts.

## Online PPO -> WorldSample/NFT reference

The current method-design reference is
`online_ppo_nft_worldsample.py`. Its ordering is deliberately different from
the legacy `m0_ppo_then_synthetic.py` diagnostic:

```text
real PPO rollout (same o0, chains, logprob/value, reward, nft_x0)
    -> 1x8 re-noise + current-policy denoise
    -> full Wan primary trajectory from that o0
    -> ResNet reward
    -> exact t+1 synthetic consumption
    -> L_PPO(real) + rho * L_NFT(synthetic)
```

`online_bridge_smoke.py` and `cosmos_wrist_adapter_smoke.py` are dependency-
light and CPU-only. They should pass before any GPU job. `--skip-wan` is a
PPO/contract fallback and intentionally does not fabricate an NFT reward.

The serial scorer is Wan-primary by default. To add the strict Cosmos-WoVR
wrist side stream from the same real `o0`, enable it with
`--cosmos-wrist-port <port>` and point `--cosmos-wrist-worldpilot-root` at the
checked-out `WorldPilot`. The canonical Goal v3 wrist checkpoint uses the
primary-camera reference layout, so the online CLI defaults to
`--cosmos-wrist-primary-reference`; use
`--no-cosmos-wrist-primary-reference` only with an older wrist-reference
checkpoint. The adapter validates the server metadata, sends `[B,8,7]`
actions, preserves the five-row context-action history, and writes a separate
`cosmos_wrist_variant_XX_full.mp4` for each synthetic branch. The primary Wan
frames still provide the Goal ResNet reward; the wrist stream is a time-aligned
conditioning/debug stream at this stage.

The distributed Ray worker and a real Wan+Cosmos GPU run are still tracked in
`refine-logs/UNCERTAINTY_LOG.md` and must be completed before a long run.

## Merge rule

Merge the RLinf-side changes as one feature with four logical parts: world
model environments, OpenPI/GRPO actor plumbing, Hydra configs/evaluation
configs, and tests/docs. Keep the project-level launcher and model-training
scripts as a separate experiment-integration change. Do not merge both
`RLinf/rlinf/envs/world_model/*` and the corresponding files from
`RLinf-kir-wrist`; choose the `RLinf` versions used by the launcher.
