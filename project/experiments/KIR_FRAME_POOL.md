# Spatial dual-camera frame pool

`kir_frame_pool.py` prepares a CPU-only, shared frame index for Wan primary
and Cosmos wrist v3. It leaves the original HDF5 data unchanged and stores
only a compressed index, provenance manifest, and resumable scan metadata.
The training launcher defaults to first-grasp starts described below; the
index commands themselves do not start training.

```bash
/data73/qianb2608/grpo-world-model-runtime/venvs/rlinf-openpi/bin/python \
  experiments/kir_frame_pool.py build

/data73/qianb2608/grpo-world-model-runtime/venvs/rlinf-openpi/bin/python \
  experiments/kir_frame_pool.py check
```

Defaults:

- Source: `/data41/qianb2608/wan/hf_home/LIBERO-Cosmos-Policy/all_episodes`;
  only `libero_spatial` episodes are indexed.
- Output: `runtime_artifacts/kir/spatial_dual_view_v1/`.
- Candidate stride: eight frames; minimum real history: four actions;
  minimum remaining recorded future: eight actions and successor frames.
- Four CPU scan workers. `--source`, `--output`, `--stride`, and `--workers`
  override these choices.

Every source RGB frame is fully decoded during construction. Camera lengths,
action dimensions/finiteness, optional proprio alignment and success labels
are validated. Transient NFS reads are retried. A failed scan stops the build
without publishing a complete manifest; rerunning reuses unchanged verified
episodes. Duplicate episode content is excluded. An existing complete pool
is checked and reused; changed settings or sources require a new output path.

## First gripper close minus 16 frames

The current training default is
`runtime_artifacts/kir/spatial_dual_view_first_close_pre16_v1/`.
Build it from the fully verified shared frame table:

```bash
/data73/qianb2608/grpo-world-model-runtime/venvs/rlinf-openpi/bin/python \
  experiments/kir_frame_pool.py select-grasp --backtrack 16

/data73/qianb2608/grpo-world-model-runtime/venvs/rlinf-openpi/bin/python \
  experiments/kir_frame_pool.py check \
  --output runtime_artifacts/kir/spatial_dual_view_first_close_pre16_v1
```

The selector detects the first negative-to-positive gripper command, with
three consecutive open commands before it and three closed commands from it
onward. A transition at `t_close` means `action[t_close]` is the first close
command, executed after `observation[t_close]`. The selected observation is
exactly `t_close - 16`, even if it does not lie on the old eight-frame grid.
Each episode contributes one initial start and at most one grasp start.
An event with insufficient history or recorded future contributes only the
initial start; it is never clamped or replaced by a later grasp attempt.

The local pool contains **500 initial + 500 pre-grasp starts**, 50 pre-grasp
starts per task, from 430 successful and 70 failed episodes. All 500 episodes
pass the history/future checks. RGB is neither duplicated nor rescanned:
the selector checks the original pool's integrity and source fingerprints,
reads the action streams, and records the original verified frame table.
`--base-pool`, `--output`, `--backtrack`, and `--stable-frames` configure a
separate immutable selection. The original grid pool is preserved.

This is a grasp-attempt heuristic, not a claim that the right object was
grasped successfully or that a failed episode failed at the grasp. The
manifest records the rule, base-pool hash, first close, requested start and
selection status for every episode. Offline `start_event_t` indices record
the close-command time (`-1` for initial starts); event/future annotations
are not included in the default reset packet or supplied to either model.

Export a few paired source videos from the selected start through the last
recorded frame, along with length statistics for all selected episodes:

```bash
/data73/qianb2608/grpo-world-model-runtime/venvs/rlinf-openpi/bin/python \
  experiments/kir_export_videos.py --tasks 0,4,7 --fps 20
```

The default output is `videos_to_end/` within the selected pool. It contains
six side-by-side MP4s (one successful and one failed episode per requested
task), `index.html`, `videos_manifest.json`, `trajectory_length_stats.json`,
and a per-episode `trajectory_lengths.csv`. Each MP4 is decoded after export
to check its frame count, playback rate, and first/last images. No tail is
dropped or padded. FPS controls playback only; source timestamps are absent.
The videos show real recordings, not imagined rollouts. Source length
reduction is not a measured reduction in GRPO training time.

## Files and time convention

`manifest.json` records source paths, sizes, mtimes, per-episode content
hashes, task labels, camera conventions, counts and the compressed index hash.
`validation.json` records the last successful source/checksum check and decoded
initial, intermediate and failed-episode examples.
`indices.npz` contains integer arrays:

| Arrays | Meaning |
| --- | --- |
| `frame_episode`, `frame_t` | One shared row per real paired camera observation |
| `action_t`, `next_t` | Action leading from this row to its successor; `-1` when there is no recorded successor |
| `start_episode`, `start_t`, `start_kind` | Reset candidates: initial `0`, intermediate `1` |

`action[t]` leads from `observation[t]` to `observation[t+1]`. When an HDF5
has N observations and N actions, its final action has no recorded successor
and is excluded from transitions. All real frames remain indexed; no common
episode horizon or future padding is introduced.

The source has no independent camera timestamps, so synchronization follows
its paired stream ordering. This index validates that convention and stream
integrity; it does not independently establish camera capture timing.

The original pool's intermediate candidates lie on a regular grid; the
selected grasp pool uses the event rule above. Neither provides per-frame
success or failure annotations. Each retains the source episode's final
success label, including failures. `sample_groups` requires an explicit
KIR fraction, samples tasks uniformly within the requested start kind and
repeats the same start ID for every branch of a GRPO group.

## Reading a reset

```python
from pathlib import Path
from kir_frame_pool import FramePool

pool = FramePool(Path("runtime_artifacts/kir/spatial_dual_view_first_close_pre16_v1"))
ids = pool.sample_groups(2, group_size=8, kir_fraction=0.5, seed=42)
reset = pool.read_start(int(ids[0, 0]))
```

The example KIR ratio is illustrative, not a claim about the paper protocol.
The default reader returns only the current policy RGB, fixed references,
past RGB and past actions. Future source images/actions are available solely
with `include_future=True` under `audit_future` for offline alignment checks.
They must never be supplied to the policy during imagined rollout.

At intermediate observation t:

| Consumer | Fixed reference / memory | Context actions |
| --- | --- | --- |
| Wan | primary[0]; primary[t-3], ..., primary[t] | four actions t-4, ..., t-1 in rows 1–4 |
| Cosmos wrist v3 | primary[0]; wrist[0], wrist[t-2], wrist[t-1], wrist[t] | three actions t-3, ..., t-1 in rows 2–4 |

Unused context rows contain zero motion and signed open-gripper `-1`. At an
initial start, memory repeats frame zero and all context actions are dummy
open-gripper actions. The actual initial reference is preserved for KIR;
it is not replaced by the selected intermediate frame.

Decoded RGB is returned in `rlinf_runtime_rot180` space, with the same
horizontal flip from released Cosmos `flipud` images as the existing video
evaluation. Raw source resolution is retained. The later environment adapter
must apply its existing model-specific resize/crop and runtime-to-model camera
conversion exactly once. Actions already use signed LIBERO gripper values;
the reader does not perform policy `[0,1]` gripper conversion or normalization.
Source proprio is validated but not passed to the stateless policy.

These episodes also supply the current local world-model metric diagnostics.
Once used for KIR training, they cannot serve as an independent held-out
evaluation set. Benchmark success should use separate LIBERO reset trials.

## Training integration

`run_spatial_wovr_8gpu.sh` uses the first-close-minus-16 pool for training resets.
`SPATIAL_KIR_FRAME_POOL` overrides the pool directory. The default configuration
sets `use_kir_frame_pool=true`, `enable_kir=true`, and `kir_fraction=0.5`.
Each GRPO group has a 50% chance of a keyframe start and a 50% chance of an
initial start. The sampler balances tasks within each start kind. The original
release used 196 keyframe files among 692 starts (28.32%).

For each task, the training environment takes the largest recorded action
count among successful source episodes and adds `success_horizon_grace_steps`
(16 by default). A reset at source frame `t` has `deadline - t` actions
remaining, capped at the usual per-episode horizon. Tasks with no successful
source episodes use the usual number of actions from reset. Starts with no remaining
actions are excluded. Success on the deadline earns the usual success reward.
An unfinished episode at the deadline is truncated with zero return, and the
actor masks subsequent actions. The shared-pool launcher skips finished slots
when generating later chunks. LIBERO evaluation uses its benchmark horizon.

The environment samples one pool start per group and repeats it for all eight
branches. It uses RLinf's existing rank-offset torch RNG and the existing
`update_reset_state_ids` / `is_start` bootstrap lifecycle. All branches share
the same task, source frame, references and histories. The current frame is
exposed to the policy; generation then advances with the policy's new actions.
Wan preserves its original reference plus four recent frames. Cosmos v3
preserves its primary reference and original wrist anchor plus three recent
wrist frames. Context actions are already signed and are not converted a
second time. The policy state remains an eight-dimensional zero placeholder.

Training does not build or advance a shadow LIBERO simulator in this mode.
Reset initialization comes from the pool, subsequent primary/wrist RGB from
their respective learned models, and rewards from the existing Wan classifier.
Independent `env.eval` remains LIBERO. `kir_initialized` and `kir_source_frame`
are included in episode metrics; reset info contains the selected start IDs.

### First-success action mask

The Spatial configuration enables `mask_after_first_success=true` with
`success_reward_threshold=0.9` and `success_consecutive_frames=1`. Sparse
rewards and success use the same inclusive probability threshold. Relative
reward scaling remains `reward_coef=5`.

The 0.9 operating point was selected by balanced accuracy on 400 whole
recordings (344 successful, 56 failed), then checked on 100 held-out
recordings (86 successful, 14 failed). Across all 500, it detects 395/430
successful recordings and flags 12/70 finally failed recordings. At 0.5,
those counts are 420/430 and 24/70. A 0.99 threshold reduces false positives
to 2/70 but only detects 153/430 successes, making training rewards sparse.
These are episode-outcome diagnostics on real recordings, not frame-level
success annotations or an evaluation of generated frames. Ordinary and KIR
starts have identical decisions at these operating points. See
`runtime_artifacts/reward_calibration/spatial_500_v1/report.json` for splits
and the complete threshold comparison.

If action 3 of an eight-action chunk first produces success, that action's
reward and policy loss remain valid; actions 4–8 and all later chunks have
zero training weight. The environment zeros post-success rewards, retains
the reward state at the terminal frame, and carries the terminal index
through RLinf's rollout tensors. The terminal suffix is true so consumers
that inspect only the final chunk slot still see a terminal episode.
There is only one terminal event per episode. The mask is independent of
metric recording and resets with the episode.

RLinf's existing actor computes `loss_mask_sum` from these action-level
terminations and normalizes each trajectory by its own valid action count.
`valid_episode_len` reports that count. `first_success_step` is one-based
relative to the reset, with `-1` for episodes without a success; it is not
the original source frame number. `episode_len` retains the rectangular
generation-step count. The video models still generate complete chunks;
this change controls training contributions, not the diffusion batch size.

This mode requires `auto_reset=false`, `ignore_terminations=false`,
action-level rewards, and per-action or per-token log probabilities. It
requires the per-frame Wan classifier rather than a terminal-only Cosmos
value. Set `mask_after_first_success=false` for the historical chunk-boundary
mask. Other launchers retain that historical behavior unless they enable
the new flag explicitly.

Run CPU/config/source checks before launching from the GPU terminal:

```bash
bash experiments/run_spatial_wovr_8gpu.sh --check
```

The check writes `kir_pool_check.json` and `kir_reset_smoke.json` in the printed
runtime directory. The latter uses the real environment constructor, reset and
chunk-step code with stub video/reward models, plus real source histories.
It also constructs the installed `RecordVideo` wrapper and checks offload,
onload and dataset lookup through the same `get_env_attr` path as EnvWorker.
The world-environment base implements Gymnasium's `get_wrapper_attr` protocol
so video recording preserves the environment's memory lifecycle methods.
It tests initial/intermediate reset shapes, one chunk, fixed anchors, group
identity and RNG replay. It does not test learned generation or a GPU GRPO
update. The next required runtime validation is an actual GPU rollout/update.
For event-selected pools it also checks that reset frames precede the stored
first-close event by the configured number of frames and records the rule.
With the first-success mask enabled, it additionally runs two CPU chunks
through the real environment and native actor preprocessing. It checks
first/middle/last-step successes, cross-chunk boundaries, threshold equality,
and failures. The diagnostic then runs native GRPO advantages and the clipped
policy loss backward: post-success rewards/gradients must be zero, and each
trajectory's total gradient contribution must follow its own valid length.
The learned models are stubbed in this check, so it does not constitute a
GPU policy update or a test of the reward classifier's accuracy.

For an initial-state-only ablation with the same offline reset path:

```bash
bash experiments/run_spatial_wovr_8gpu.sh --check env.train.enable_kir=false
```

This keeps all intermediate starts out of the sampler. To run the historical
simulator-backed bootstrap instead, set `env.train.use_kir_frame_pool=false`,
`env.train.use_libero_state_bridge=true`, and `env.train.enable_kir=false`.
Pool mode rejects simulator/proprio/value/cross-view combinations that cannot
be synchronized to the stored source state. Automatic resets are disabled so
GRPO groups and the existing post-success mask remain intact.
