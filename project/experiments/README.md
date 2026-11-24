# Selected experiment entry points

This directory contains launchers and utilities for the Wan-primary +
Cosmos-WoVR-wrist GRPO line and LIBERO-Spatial GRPO in the simulator. The full
local workspace contains many unrelated experiments and is not copied here.

Before publishing, replace machine-specific paths in the shell scripts with
environment variables such as `PROJECT_ROOT`, `DATA_ROOT`, `MODEL_ROOT`, and
`COSMOS_PYTHON`. Checkpoints, datasets, logs, videos, and generated reports
remain outside this repository.

## Spatial world-model data collection and updates

See [WORLD_MODEL_UPDATE.md](WORLD_MODEL_UPDATE.md) for the fixed-policy paired
LIBERO collector, shared train/validation manifest, concurrent Wan/Cosmos updates
on two or four GPUs each, complete resume checkpoints, and old/new model acceptance.
`run_spatial_wm_update_4gpu.sh check --check-runtime` is CPU-only. Collection,
backward probes, and training are explicit commands and do not modify an active
GRPO run. `run_spatial_wm_update_8gpu.sh` uses eight collectors and concurrent
four-GPU model groups, with four accumulation microsteps for batch 16 per model.
`queue_spatial_wm_update.py` can wait for a pinned GRPO run to finish successfully
at step 20, then run the collection, backward probe, updates and pair packaging.
The newly added recipe and memory paths still need the documented GPU probe.

Fresh Cosmos updates now use a versioned reconstruction of the wrist-v3 recipe:
fixed wrist anchor, causal actions, train-only native image augmentation,
memory corruption at 0.5, static futures at 0.05, and the configured weight decay.
The unavailable original corruption/static algorithms are explicitly marked as
reconstructed. `--cosmos-recipe auto` preserves the original recipe when resuming
an existing optimizer checkpoint. See `WORLD_MODEL_UPDATE.md` for initialization,
effective settings, and the limits of the original release metadata.

World-model training now retains reusable CUDA memory by default, with one
allocator owner per GPU and 3.5 GiB device headroom. The explicit entry points
are `run_spatial_wm_update_reserved_4gpu.sh` and
`run_spatial_wm_update_reserved_8gpu.sh`. See the memory-reservation subsection
in `WORLD_MODEL_UPDATE.md` for flags, logs and the remaining GPU validation.

## Spatial dual-camera KIR frame index

See [KIR_FRAME_POOL.md](KIR_FRAME_POOL.md) for the shared primary/wrist/action
frame index and reset reader. `kir_frame_pool.py build` validates existing
Spatial HDF5 episodes and writes compact indices under
`runtime_artifacts/kir/spatial_dual_view_v1/` without copying RGB data. It
preserves real frame ordering, fixed anchors and separate Wan/Cosmos v3
histories. `kir_frame_pool.py select-grasp --backtrack 16` then selects one
start exactly 16 frames before each episode's first stable gripper close.
The dual-view training launcher defaults to this selected pool at
`runtime_artifacts/kir/spatial_dual_view_first_close_pre16_v1/`.
The data preparation commands do not start GRPO.

## Spatial world-model video metrics

See [WORLD_MODEL_EVAL.md](WORLD_MODEL_EVAL.md) for the Wan primary-view and
Cosmos wrist v3 evaluation. It reuses 500 local Spatial RGB/action episodes,
retains each episode's available complete eight-action chunks, and reports
FPS, LPIPS, FID, variable-length FVD and FloLPIPS. The entry point is
`run_world_model_eval.sh`; metric dependencies are installed separately with
`scripts/setup_world_model_metrics.sh`. GPU inference and optical-flow CUDA
validation must run from the GPU-enabled terminal.

## LIBERO-Spatial on eight GPUs

`run_spatial_grpo_8gpu.sh` loads the Pi05 step-12000 checkpoint. It uses the
RLinf Spatial GRPO settings with the previous LIBERO-10 batch configuration:
micro batch 32 per GPU, global batch 2048, and 8 gradient accumulation steps.
Training duration remains at 1000 epochs, with no explicit step limit. Check
assets and resolve the configuration with:

```bash
bash experiments/run_spatial_grpo_8gpu.sh --check
```

Start training with:

```bash
bash experiments/run_spatial_grpo_8gpu.sh
```

The local Actor subclass warms PyTorch's native caching allocator after RLinf
initializes the rollout and simulator workers. It targets up to 80% of device
capacity in the Actor's allocator, including model and optimizer tensors, and
limits new reservations to leave at least 6 GiB globally free at that moment.
It delegates the Actor's usual `empty_cache()` calls without returning cached
blocks to the driver. The temporary reservation tensor is deleted immediately;
subsequent training allocations reuse the cached blocks in the same process.
There is no background process holding tensors that compete with training.

Only Actor workers retain their cache. Rollout and simulator allocators are
unchanged, and the RLinf source tree is not modified. PyTorch can still release
cache internally under allocation pressure; this is best-effort retention,
not an exclusive GPU reservation. Existing allocations from other programs
cannot be reclaimed. A racing allocation that prevents warming the cache emits
a warning and does not abort the job.

Inspect `gpu_preflight.log` for GPU UUIDs and resident processes at startup.
Each Actor also logs `[actor-memory-pool rank=...]` with allocated/reserved
memory and global free memory. Adjust or disable retention with Hydra overrides:

```bash
bash experiments/run_spatial_grpo_8gpu.sh --check actor.memory_pool.target_fraction=0.75
bash experiments/run_spatial_grpo_8gpu.sh actor.memory_pool.enabled=false
```

Hydra overrides are passed to both configuration resolution and training, and
`resolved_config.yaml` records their effective values. Runs are written under
`/data73/qianb2608/grpo_runs/spatial_pi05_12000_grpo_8gpu/` unless
`SPATIAL_GRPO_OUTPUT` is set.

If stopping the launcher leaves Ray workers or GPU allocations behind, inspect
one specific run with the standard-library-only cleanup utility:

```bash
python3 experiments/stop_spatial_grpo.py --run-dir /path/to/the/run
python3 experiments/stop_spatial_grpo.py --run-dir /path/to/the/run --apply
```

The first command only lists matches. `--apply` sends TERM, waits ten seconds,
then sends KILL to remaining verified processes. Selection uses the run's
exported output directory, unique Ray directory, and child relationships;
it checks ownership and process start time before signaling. Other run IDs
and the calling shell are excluded. Logs and checkpoints are retained. Run
it in the GPU terminal, where the host workers are visible. It does not use
a global `ray stop`, GPU reset, or a broad process-name kill.

CPU tests for reservation budgeting and allocator reuse:

```bash
python3 -m unittest discover -s experiments/tests -v
```

## Spatial Wan + Cosmos wrist on eight GPUs

`run_spatial_wovr_8gpu.sh` is the separate dual-view world-model entry point.
It uses the selected native Pi05 step-12000 policy, Wan Spatial primary
frames/reward model, and `Cosmos-WoVR-Wrist-LIBERO-Spatial-v3-step20000` wrist
frames. Training resets use the unified offline Spatial frame pool. A group of
eight branches shares the same source start; intermediate starts are sampled
with probability 196/692, matching the released Spatial pool's file ratio.
The stateless policy receives a zero state placeholder. Training no longer
builds or advances the LIBERO shadow simulator in this configuration.
Training rewards come from Wan's learned reward model; evaluate the resulting
policy separately in real LIBERO to obtain benchmark success rates.
The Spatial preset now enables a first-success action mask: the first
successful action remains valid, while subsequent rewards and gradients are
zeroed. Both sparse rewards and termination use probability `>=0.5`.
The native actor derives each trajectory's valid-length normalization from
these per-action terminal signals; see [KIR_FRAME_POOL.md](KIR_FRAME_POOL.md).

Prepare and check the code while the wrist download is incomplete:

```bash
bash experiments/run_spatial_wovr_8gpu.sh --check
```

After the download finishes, check complete DCP shards and SHA-256 values:

```bash
bash experiments/run_spatial_wovr_8gpu.sh --check-assets
```

Neither check starts CUDA workers, Cosmos servers, or training. Launch only
when ready, from the terminal with GPU access:

```bash
screen -S spatial-wovr8
bash experiments/run_spatial_wovr_8gpu.sh
```

Detach using Ctrl+A then D. The coordinator terminates its own services when
training exits or receives Ctrl+C/SIGTERM. It uses a private Ray cluster;
it never runs a machine-wide `ray stop`. Watch the printed output directory's
`train.log` and `cosmos_rank0.log` through `cosmos_rank7.log`.

For the selected 20-step production run, use the dedicated entry point:

```bash
bash experiments/run_spatial_wovr_20step.sh --check
screen -dmS spatial-wovr20 bash experiments/run_spatial_wovr_20step.sh
screen -r spatial-wovr20
```

This keeps the full rollout settings and micro batch 32, with eight gradient
accumulations. It saves policy weights and runs 500 real LIBERO Spatial trials
at steps 5, 10, 15 and 20. The complete optimizer/scheduler resume state is
saved at steps 10 and 20; steps 5 and 15 retain policy weights only.
Outputs default to
`/data12/qianb2608/grpo_runs/spatial_wovr20_pi05_12000_8gpu/<timestamp_pid>/`.
This starts from the selected Pi05 step-12000 policy; the backward peak test
did not save a policy checkpoint. The general entry point still defaults to
50 steps. Run the screen command once, in the terminal with GPU access.

The default deployment colocates actor, rollout, Wan environment, and one
independent Cosmos wrist service on each of GPUs 0–7. Environment rank N uses
port `18059+N`. Set `SPATIAL_COSMOS_PORT` for a different port range and
`SPATIAL_WOVR_GPUS` for a different ordered list of eight visible GPU IDs.
An occupied port or an incompatible checkpoint/service is rejected.

The policy retains its native 10-action prediction horizon and executes eight
actions per chunk, as required by the world models. The v3 wrist route uses
the primary view in the reference slot, without the v4 cross-view history
adapter. Camera space is `rlinf_runtime_rot180`; the Pi05 `[0,1]` gripper is
converted to the signed LIBERO action exactly once. The custom wrist model
does not supply a value or proprioception head.

Micro batch is 32 per GPU, gradient accumulation is 8, and global batch is
2048. The default has 128 environments and eight rollout epochs of 240
actions: `128 * 8 * (240 / 8) = 30720` chunk samples, exactly 15 global
batches. With 64 environments the last batch would be incomplete. Training
uses `max_epochs=50`, `max_steps=50`, saving and evaluating every five GRPO
outer steps. Each outer step contains 15 global optimizer batches with the
current rollout settings. Cosmos requests are split into batches of two images.
Wan offloads between phases. The actor retains reusable CUDA cache during
updates and releases inactive cache at weight synchronization before the next
world-model rollout; the simulator-only cache settings cannot be applied
unchanged to the extra world-model processes. GPU peak usage and throughput
still require verification on the actual eight GPUs.

At steps 5, 10, ..., 50 the runner saves the policy weights,
synchronizes the updated policy, then evaluates in the real LIBERO Spatial
simulator. Each evaluation uses all ten tasks and the same first 50 distinct
initial states per task. Success comes from LIBERO's simulator task checker.
There is no reward-ResNet threshold in this real evaluation. Eight ranks use
64 concurrent environments, automatic resets at true success or the official
240-action limit, and eight rollout waves. This covers all 500 trials even
when every episode times out; exhausted slots stop producing counted trials.
Both environment and rollout workers share this evaluation budget.

Each saved step gets `evaluations/global_step_N/metrics.json` and
`episodes.csv` under the experiment output, containing aggregate and per-task
rates plus all 500 task/trial/outcome rows. The runner rejects missing,
duplicate or extra trials rather than reporting an incomplete rate. Eval
metrics are logged at the actual completed GRPO step N. A failed evaluation
leaves the preceding policy weights available for standalone testing or policy
initialization.

The default output root is now `/data12/qianb2608/grpo_runs/` because `/data73`
had only about 48 GiB free when this setup was checked. Full model weights
are retained at all ten save points; the resumable optimizer/scheduler state
is saved at step 50 only (`actor.fsdp_config.save_training_state_at_steps`).
Steps 5–45 provide policy weights for evaluation or policy initialization;
exact optimizer-state resume requires the complete step-50 checkpoint.
This avoids keeping ten copies of the large optimizer state. Explicit
`SPATIAL_WOVR_OUTPUT` still selects a different output directory.

New launches include additional training diagnostics without changing the
native rewards, masks, loss normalization, backward calls or optimizer steps:

| Metric | Meaning |
| --- | --- |
| `rollout/grpo_groups_total` | GRPO groups collected across all eight actor ranks |
| `rollout/grpo_groups_kept` | Groups with actions retained by native reward filtering |
| `rollout/grpo_groups_signal` | Retained groups with nonconstant trajectory scores |
| `rollout/grpo_group_keep_frac`, `rollout/grpo_group_signal_frac` | Corresponding fractions of all collected groups |
| `actor/micro_count`, `actor/empty_micro_count`, `actor/empty_micro_frac` | Executed micro-batches and those with no valid training actions |
| `actor/valid_actions`, `actor/valid_action_frac` | Valid action count and fraction across the executed micro-batches |
| `actor/ratio_valid`, `actor/ratio_abs_valid`, `actor/clipped_ratio_valid` | Native ratio metrics weighted by valid action count across micro-batches and ranks |
| `env/first_success_mean`, `env/first_success_count` | Mean first-success action index and count, excluding failed episodes' `-1` sentinel |

Counts and weighted numerators are summed across ranks before ratios are
computed. The fixed seven log-probability dimensions per action make action
weighting equivalent to token weighting for this Spatial policy. A rank or
micro-batch with no valid actions does not dilute the new ratio metrics.
If the complete update has no valid actions, valid-only ratios are `NaN`;
the successful-frame mean is also `NaN` when no episode succeeds. Group counts
are measured once per outer rollout; micro-batch counts include all optimizer
batches and update epochs in that outer step. The original metric keys remain
available, including the legacy `first_success_step` mean that includes `-1`.
Already running workers retain their loaded code; these diagnostics take
effect on the next launch.
CPU regression checks cover uneven rank weights, empty ranks and micro-batches,
native reward-filter compatibility, and unchanged gradients/parameter updates.
The two-process Gloo test skips when local sockets are unavailable; run it in
the GPU terminal with the launcher's runtime `PYTHONPATH` to exercise the real
collective. CPU checks do not validate NCCL behavior on the eight GPUs.

An independent email watchdog can attach to an already running 20-step job.
Configure the SMTP sender and recipient in the GPU terminal first:

```bash
python3 experiments/watch_training_email.py --configure
```

The authorization code is entered through a hidden terminal prompt and kept
in a mode-600 local file under `/data41/qianb2608/dyk/.config/grpo-notify/`.
It is not placed in command arguments, logs or notification messages. Setup
sends a test email; confirm receipt in the destination mailbox. A different
existing JSON configuration can be selected with `--smtp-config /path/to/smtp.json`.
Its fields are `host`, `port`, `security` (`ssl`, `starttls` or an explicitly
selected `plain` relay), `username`, `from`, `to` (a list of addresses), and
`password_file` or `password_env` when authentication is required.
Optional `auth_method` is `auto` (Python's default negotiation) or `login`
(explicit two-challenge AUTH LOGIN without an initial response, requiring
`ssl` or `starttls`). The local QQ configuration uses `login` to investigate
a disconnect during automatic authentication; this is not confirmation that
the authorization code is valid or that delivery works.

The existing `dyk/email/watchdog_mail.json` configuration can be reused through
the default config symlink. Test delivery from the project directory:

```bash
cd /data41/qianb2608/dyk/grpo-world-model
python3 experiments/watch_training_email.py --test-email
```

This prints connection/TLS/greeting, EHLO, login and message-submission stages.
On failure, the error contains the failing stage, exception type and numeric
status code when available. It never prints authorization data or raw server
responses. `connect_ssl_and_greeting` covers DNS, TCP, the verified TLS handshake
and the initial SMTP greeting; it does not distinguish those operations by
itself. `--diagnose-smtp` checks the configured transport and EHLO without
logging in or sending mail. Neither command starts the watchdog. A disconnect
alone does not establish that the authorization code is invalid.
Known advertised authentication mechanisms are shown without raw EHLO data.
With explicit LOGIN, `auth_step=auth_command`, `username_response` or
`authorization_response` identifies the command/response awaiting a reply when
the error occurred. Neither response values nor their base64 encodings are
printed. No alternative authentication is retried after a disconnect.
The password-file reader ignores leading/trailing blank lines from editing,
preserves spaces and rejects multiple nonempty lines without displaying them.

An explicit SMTP 535 response means authentication was rejected. Check the
configured mailbox's SMTP service and generate an authorization code in its
account settings. Replace the local code without re-entering the endpoint or
recipient:

```bash
python3 experiments/watch_training_email.py --update-auth-code
```

The prompt requires hidden terminal input; do not place the code in a command
argument or chat message. It updates the existing `password_file` atomically
with mode 600, then sends one test email. It does not start the watchdog.

Start the watcher once, in its own screen on the same GPU host:

```bash
screen -dmS spatial-watchdog bash experiments/run_spatial_wovr_watchdog.sh
screen -r spatial-watchdog
```

The wrapper attaches to the newest initialized run under
`/data12/qianb2608/grpo_runs/spatial_wovr20_pi05_12000_8gpu/`. To select an
existing run explicitly, set `SPATIAL_WATCH_RUN=/absolute/output/path` before
running it. Training is not restarted. Look for `WATCHDOG ARMED` in the screen
or its durable `runtime_artifacts/training_watchdog/launcher_*.log`.

Every two seconds it checks the recorded exit code and the identities of
this run's launcher, learner and eight Cosmos services. Process start times
protect against PID reuse. A quiet log is not considered a failure. Normal
completion, a nonzero exit or disappearance of a tracked process triggers an
email with the run directory, exit code when available, process roles and
training log path. Detection is normally within the polling interval; email
arrival also depends on SMTP and the destination mailbox. Failed sends are
persisted and retried with 5-to-60-second backoff. One local lock prevents
duplicate watchers, and a sent-state marker avoids routine repeat notices.
An SMTP acknowledgement lost during a crash can still cause a retry email.
Heartbeat and delivery-state writes are best effort. If the state filesystem
fills up, an already running watcher keeps checking processes and attempts
the email using in-memory state; state/logging errors no longer abort delivery.
State-write warnings are throttled to once per minute and heartbeat write
attempts remain every 30 seconds. A broken logging pipe is also tolerated.
If the sent marker cannot be saved, restarting can send a duplicate notice.
Starting a new watcher still requires an accessible lock/config directory.

Per-run `state.json` records `watching`, `delivery_pending` or `sent`. The
watcher does not stop/restart training or delete files. An isolated agent
session without host PID visibility cannot arm it. This watcher must remain
running; a whole-host shutdown requires monitoring from another machine.
The run's `source_paths.json` records its GPU count; four-GPU runs track four
Cosmos services plus the launcher and learner. Legacy runs default to eight
services. `SPATIAL_WATCH_LOG_ROOT` can relocate launcher logs, and
`SPATIAL_WATCH_RUN_ROOT` changes the directory used by `--latest-run-root`.
CPU tests use mock SMTP transports and do not establish actual mail delivery:

```bash
python3 -m unittest discover -s experiments/tests -p test_watch_training_email.py -v
```

The four-GPU entry points default to physical devices `0,1,2,3`. They use the
same base policy, group size 8, optimizer settings, global batch 2048 and
30,720 rollout chunks per full training step as the eight-GPU run:

| Setting | Eight GPUs | Four GPUs |
| --- | ---: | ---: |
| Training environments | 128 | 64 |
| Rollout epochs | 8 | 16 |
| Environments per GPU | 16 | 16 |
| Micro batch per GPU | 32 | 32 |
| Gradient accumulation | 8 | 16 |
| Concurrent real evaluation environments | 64 | 32 |
| Worst-case evaluation waves for 500 trials | 8 | 16 |

Run a single optimizer-update memory probe in the GPU terminal first:

```bash
screen -dmS spatial-backward4 bash experiments/run_spatial_backward_peak_4gpu.sh
```

This probe collects 64 environments x 2 rollouts x 16 action chunks = 2048
samples, with 16 micro backward calls per rank. It disables save/evaluation
and writes four actor memory reports under
`/data12/qianb2608/grpo_runs/spatial_backward_peak_4gpu/`.
It does not measure the extra GPU contexts used by real LIBERO evaluation.

The full four-GPU 20-step run retains policy saves and real 10x50 Spatial
evaluations at steps 5, 10, 15 and 20, and full resume state at steps 10 and 20:

```bash
screen -dmS spatial-wovr20-4 bash experiments/run_spatial_wovr_20step_4gpu.sh
```

The four-GPU 20-step entry point also starts a device-memory monitor at a
0.2-second sampling interval. It covers rollout, updates, saving and real
evaluation until the launcher exits, which also stops the monitor. Results
are written to `gpu_peak/device/summary.json` and `gpu_peak/device/gpu_peaks.csv`
inside the training output directory. This device monitor does not enable
the actor's probe-only allocator counters.

If `/tmp` is nearly full, set `SPATIAL_WOVR_TMP_ROOT` to a short directory on a
filesystem with sufficient free space before starting the job, for example:

```bash
export SPATIAL_WOVR_TMP_ROOT=/data41/qianb2608/dyk/r
screen -dmS spatial-wovr20-4 bash experiments/run_spatial_wovr_20step_4gpu.sh
```

The launcher creates a separate `w-XXXXXXXX` directory there and passes it as
both `TMPDIR` and `RAY_TMPDIR`; the printed path identifies this run's temporary
files. The default remains `/tmp/spatial-wovr8-XXXXXXXX` when this variable is
unset. Ray adds its session and socket names, so the launcher rejects custom
roots whose estimated Unix socket path would reach 108 bytes. Use a short
absolute path such as the example above rather than a project subdirectory.
The temporary directory is retained for diagnosis after the run.

Once that run has printed its initialized training log path, attach a watchdog
to the new four-GPU run in the same host terminal:

```bash
screen -dmS spatial-watchdog4 bash experiments/run_spatial_wovr_watchdog_4gpu.sh
```

Full runs are stored under
`/data12/qianb2608/grpo_runs/spatial_wovr20_pi05_12000_4gpu/`. The training
launcher accepts `SPATIAL_WOVR_GPUS` to select another set of four physical
devices; logical worker placement stays `0-3`. No GPU jobs are started by
the shell/config regression tests.

### Opt-in four-GPU throughput entry points

The optimized entry points are separate from the ordinary launchers. They do
not alter a running experiment or its generated code. Both world models remain
serial: the Wan/Cosmos parallel-generation optimization is not enabled.

| Setting | Ordinary entry | Optimized entry |
| --- | ---: | ---: |
| Cosmos request batch | 2 | 4 |
| Future-frame selection / redundant frame-copy reduction | Off | On |
| Training video recording interval | Every episode batch | Every 4 episode batches |
| Generation for branches after first success | Continue | Freeze the last observation |
| Real evaluation recording interval | Every episode batch | Every episode batch |

With interval 4, each worker records its first, fifth, ninth, ... episode
batches, retaining their original `0.mp4`, `4.mp4`, `8.mp4`, ... indices.
Unrecorded batches skip frame extraction, montage assembly and MP4 encoding.

Wan compacts unfinished slots for generation while retaining each original
slot's diffusion noise. Cosmos retains the original request size, seed,
conditional VAE encoding and native solver noise, and compacts only its
denoiser and future VAE decoding. Results return to the original environment
order. A successful branch keeps the last observation from its success chunk;
later action rewards and gradients remain masked. GRPO group membership and
trajectory lengths retain their original layout. Cosmos advertises active-mask
support only for the inspected model and sampler protocol. This path requires
strict v3 wrist, first-success
masking, manual resets, and no simulator bridge, policy state, predicted
proprioception/value head or cross-view history. Unsupported combinations fail
configuration validation instead of silently using a different state path.

The transfer path caches Wan's fixed reference image until reset and builds
its uint8 clip from frames already on the CPU. Cosmos still decodes all 13
causal frames, then selects the eight future frames on the GPU before copying
them to the CPU. This reduces that decoded-frame transfer by 38.5%; it is not
a measured reduction in total rollout time.

Run the matching one-update peak probe before choosing the optimized entry
for a production run:

```bash
bash experiments/run_spatial_backward_peak_fast_4gpu.sh
```

This still collects exactly 2048 chunk samples and executes 16 backward calls
per rank. Like the ordinary probe, it does not include periodic real-evaluation
GPU contexts. After checking its output, the separate 20-step entry is:

```bash
bash experiments/run_spatial_wovr_20step_fast_4gpu.sh
```

The optimized full run keeps 64 environments, 16 rollout epochs, 240 actions,
five denoising steps for each world model, group size 8 and global batch 2048.
It retains 30,720 chunk slots and 15 optimizer batches per outer step, saves
and evaluates at steps 5/10/15/20, and saves full continuation state at 10/20.
The active-branch optimization reduces generated frames after terminal success;
it does not remove valid training prefixes or lower the nominal sample budget.
Batch-size changes can change diffusion noise grouping, so bitwise matching
against the ordinary run is not expected. No GPU throughput gain or peak-memory
reduction has been established by the CPU regression tests.

Optimized outputs default to `grpo_runs/spatial_wovr20_fast_pi05_12000_4gpu/`
or `grpo_runs/spatial_backward_peak_fast_4gpu/` under the project directory.
Generated code uses `runtime_artifacts/spatial_wovr_fast_4gpu/`. Existing
`SPATIAL_WOVR_OUTPUT`, `SPATIAL_WOVR_RUNTIME`, `SPATIAL_WOVR_GPUS` and
`SPATIAL_WOVR_TMP_ROOT` overrides still apply. `SPATIAL_WOVR_FAST_COSMOS_BATCH`
and `SPATIAL_WOVR_FAST_VIDEO_STRIDE` default to 4; trailing Hydra overrides take
precedence over wrapper defaults. `SPATIAL_WOVR_FAST_TRANSFER=1` also enables
frame selection in the separately launched Cosmos service.
The two optimized entries default their Ray/temporary root to `../r` beside
the project (`/data41/qianb2608/dyk/r` here) instead of the nearly full `/tmp`.
An explicitly supplied `SPATIAL_WOVR_TMP_ROOT` takes precedence and is checked
against Ray's Unix socket path limit.

For a speed comparison, use the same source starts, action sequence, frame
shape, denoising count and hardware. Warm both variants, then measure at least
20 chunks with mean/P95 latency, GPU peak memory and visual/output checks.
Training a continuation should use a complete optimizer-state checkpoint; the
step-5/15 policy-only saves cannot provide exact training-state continuation.
The existing formal run remains the ordinary baseline until the candidate
passes a CUDA rollout/backward check and a complete deployment memory check.

Attach the existing four-GPU email watchdog to the optimized output explicitly
after the initialized `train.log` path is printed, because its default scan
root selects ordinary runs:

```bash
SPATIAL_WATCH_RUN=/absolute/path/to/optimized/output \
  bash experiments/run_spatial_wovr_watchdog_4gpu.sh
```

The checkout omitted source directories named `env` and `datasets`. To keep
the installed RLinf backend, actor, and native Pi05 loader consistent, each
launch creates an isolated source-link tree under
`runtime_artifacts/spatial_wovr_8gpu/`: installed RLinf plus this checkout's
world-model environments and action boundary; this checkout's Cosmos source
plus missing upstream dataset modules from the installed Cosmos checkout.
The per-launch runner copy also waits for `env.interact()` to finish after
rollout completion, before advantages or actor updates. The environment sends
trajectories before offloading Wan, so receiving trajectories alone does not
guarantee that its CUDA memory is free. The launcher rejects training-pipeline
or environment-bootstrap overlap for this colocated world-model setup.
The generated runner path and SHA-256 are recorded in `source_paths.json`;
external RLinf source is not edited.
The external trees are read only. `source_paths.json`, `resolved_config.yaml`,
and `asset_check.json` record the effective setup. These machine-local
dependencies must be supplied when moving the launcher to another host.
Temporal training datasets and the optional online Reason1 text encoder are
resolved lazily; cached T5 wrist inference does not instantiate them.
`COSMOS_LOCAL_INFERENCE_CHECKPOINT` prevents experiment registration from
fetching unrelated training initialization weights when a full local model
is selected. The Cosmos server also accepts `--cosmos_vae` to select the
existing local VAE directly.

Useful path overrides are `RLINF_BASE_ROOT`, `RLINF_ENV`, `COSMOS_BASE_ROOT`,
`COSMOS_PYTHON`, `PI05_CHECKPOINT_ROOT`, `WAN_SPATIAL_CHECKPOINT`,
`COSMOS_WRIST_CHECKPOINT` (the `iter_000020000` directory), `COSMOS_VAE`,
`COSMOS_DATASET_STATS`, `COSMOS_T5_EMBEDDINGS`, and `SPATIAL_WOVR_OUTPUT`.
The wrist default points at the active download on `/data12`, and the VAE,
statistics and T5 cache use the existing `/data41/qianb2608/dyk/cosmos` links.
No checkpoint is downloaded by this launcher. Both inference and training
run with Hugging Face offline mode enabled. Hydra overrides are accepted and
recorded, for example:

```bash
bash experiments/run_spatial_wovr_8gpu.sh --check actor.memory_pool.enabled=false
```

Every launch now saves its console output in the printed runtime directory's
`launcher.log`, including errors before a training output directory exists.
If a screen session exits immediately, inspect this file and the training
output's `train.log`. Worker-failure cleanup resolves the current private
Ray cluster explicitly, so other active Ray instances do not make cleanup's
actor-discovery address ambiguous. No machine-wide Ray shutdown is used.
Cleanup also scopes the State API's `RAY_API_SERVER_ADDRESS` environment
override to that query: Ray otherwise overrides an explicit GCS address with
the launcher's `RAY_ADDRESS=local` and can rediscover unrelated clusters.

To measure memory, start this read-only monitor in a second GPU terminal
before launching training:

```bash
python3 experiments/monitor_gpu_memory.py --interval 0.5
```

It queries physical GPUs 0–7 without importing PyTorch or allocating CUDA
memory, prints observed peaks every ten seconds, and saves updates under
`runtime_artifacts/gpu_memory/<timestamp_pid>/`. Ctrl+C stops only the monitor.
Use `--duration 900` for a fifteen-minute recording, `--gpus 0,5` to select
physical devices, or `--gpus all` for every device. An existing output can be
attributed with `--run-dir /path/to/SPATIAL_WOVR_OUTPUT`: process membership
uses the inherited environment variable when readable, otherwise stays
unknown. No process command lines or environment contents are saved.

`gpu_peaks.csv` shows each card's first sample, peak usage in MiB/GiB, minimum
free memory and peak timestamp. `summary.json` also retains per-process
peaks and the process snapshot at each card's peak. `gpu_samples.csv` and
`process_samples.csv` contain the full series. Concurrent process totals are
computed within each sample, rather than summing maxima from different times.
Other jobs and CUDA allocator caches count toward device occupancy; the
first-sample delta is not isolated training memory. Sampling starts when the
monitor runs and cannot recover earlier peaks or guarantee capture of bursts
shorter than the sampling interval. Cover model startup, GRPO updates,
checkpoint saves and real evaluation before assessing the complete workload.

For a single eight-GPU rollout/update cycle, use:

```bash
bash experiments/run_spatial_wovr_smoke.sh --check
# From the terminal with GPU access:
screen -S spatial-wovr-smoke
bash experiments/run_spatial_wovr_smoke.sh
```

This keeps 128 environments, groups of eight, micro batch 32, eight gradient
accumulations and global batch 2048. It collects one rollout of 128 actions
instead of eight rollouts of 240: `128 * (128 / 8) = 2048` chunk samples,
one global minibatch and one outer GRPO cycle. The episode limit stays 240.
It saves the final policy weights and exits, with the same service cleanup as
the full launcher. The output defaults to `/data12` to avoid storing model
and optimizer checkpoints on the nearly full project volume. Inspect
`actor/grad_norm`, reward/group statistics and exit status: completing a
cycle with zero advantages verifies the pipeline but does not demonstrate
a useful policy update. This smoke does not run a 500-episode policy eval.

For an allocator peak test with real forward/backward and one native optimizer
step, use this separate entry point after stopping any training already using
the same eight GPUs:

```bash
bash experiments/run_spatial_backward_peak.sh --check
screen -dmS spatial-backward-peak bash experiments/run_spatial_backward_peak.sh
screen -r spatial-backward-peak
```

It initializes the same actor/rollout/Wan/Cosmos setup and collects one smaller
rollout: 128 environments, 128 executed actions, 2048 chunk samples. Each rank
executes eight real native micro-batch forward/backward calls (micro batch 32)
and one optimizer step. Evaluation and checkpoint saving are disabled, and
the job exits after this one cycle. It starts a read-only device-memory monitor
automatically; a second monitor command is not required.

Inside the actor, CUDA peak counters are reset after the pre-training cache
reservation and read before the post-training reservation. This excludes the
temporary cache-warming tensor from the allocated-tensor measurement. Each
backward and the complete optimizer update have separate recorded peaks.
Reserved memory still includes the existing allocator cache. These exact
allocator counters cover the actor process; the device monitor includes the
other resident models and jobs, with a 0.2-second sampling interval.

The printed output directory defaults to
`/data12/qianb2608/grpo_runs/spatial_backward_peak/<timestamp_pid>/`.
`gpu_peak/actor_rank_N.json` records each rank's native backward-call count,
optimizer-step counter, gradient norm and allocator peaks.
`gpu_peak/peak_summary.json` summarizes all eight ranks and includes the device
monitor's results. `train.log` prints `[backward-peak ...]` and `[update-peak ...]`
lines while executing. OOM or incomplete updates are labeled failed, and
available measurements are retained; an OOM record is not a measurement of
the memory that would have been required to finish successfully. Zero-gradient
updates are identified separately. This test measures one update with a smaller
rollout buffer; later iterations, full rollout buffering, checkpointing and
real evaluation require their own complete-workload measurements.

The released Spatial reward ResNet can be calibrated without loading either
world model or the policy:

```bash
/data73/qianb2608/grpo-world-model-runtime/venvs/rlinf-openpi/bin/python \
  experiments/calibrate_spatial_reward.py --device auto
```

The default scores all 500 recordings in the verified pre-grasp pool,
including every RGB frame and the unmatched action tails' recorded images.
Only primary-camera RGB enters the reward classifier. Source images undergo
the same camera conversion and `[-1,1]` normalization as the Spatial training
configuration; the reset image is excluded from each success decision.
Whole episodes are split 80/20 by task and final outcome, with seed 42.
A global threshold is selected on the calibration portion using a 0.005
grid and balanced accuracy, averaging initial and KIR starts. A second
candidate maximizes recall subject to at most 5% observed failure-to-success
errors; use `--criterion low_fpr` to make that the reported recommendation.
The independent validation portion never selects the threshold; an infeasible
5% budget is reported explicitly. The completed 500-episode scan selects 0.9
by balanced accuracy, and that is now the Spatial training default. It detects
395/430 true successful recordings, with 12/70 finally failed recordings also
flagged; the held-out counts are 79/86 and 2/14. A 0.99 threshold detects only
153/430 successes, so a low false-positive budget has a large recall cost.

`runtime_artifacts/reward_calibration/spatial_500_v1/` contains per-frame
probability caches, `episodes.csv`, `threshold_comparison.csv`, the detailed
`threshold_sweep.csv`, `per_task.csv`, and `report.json`. Existing caches are
reused only when the pool, checkpoint, model source and preprocessing match.
Both the ordinary any-future-frame rule and KIR suffix are reported, with
final-frame scores as a separate diagnostic. Failed-episode counts are small;
the report includes Wilson intervals for the observed false-positive rates.
Source final-outcome labels cannot identify the true first success frame or
distinguish an earlier transient success from a classifier error. These
recordings are not established as independent of model training data, and
calibration on real images does not verify reward behavior on generated
frames. The script reports a proposed Hydra override without modifying the
training configuration automatically.

The actor-owned Wan implementation is opt-in and has a separate entry point:

```bash
# CPU-only configuration, asset and import checks. Starts no GPU workers.
bash experiments/run_spatial_wovr_shared_pool_8gpu.sh --check-assets
# Prepare the real two-update validation without starting it:
bash experiments/run_spatial_wovr_shared_pool_probe_8gpu.sh --check
```

In this mode each actor owns its local Wan DiT/VAE and serves the colocated
environment through an authenticated local Unix socket. Only CPU image and
action arrays cross that socket. The environment continues to own the reward
classifier and trajectory state; Cosmos and policy rollout stay in their own
processes. Wan inference and the actor use the same process's native CUDA
caching allocator and default stream. Wan weights move to CPU before actor
updates; the freed CUDA pages remain cached for the actor, and its update
buffers become reusable by Wan on the next rollout. Actor updates, checkpoint
operations and model synchronization are serialized against Wan requests.
An update before Wan acknowledges offload fails explicitly.

The `actor.shared_wan_pool.target_job_gib=75` target means the whole task's
resident memory on each device, including Cosmos/rollout/environment workers.
It does not mean 75 GiB of actor tensors. The shared allocator starts with a
65 GiB whole-task target, leaving space for the first policy and Cosmos
inferences. After the first complete dual-view chunk, it switches to 75 GiB.
Its NVIDIA XML inventory counts both CUDA and graphics-only EGL processes,
deduplicating processes reported in both categories. It tops up its
cache around the other task processes and retains at least 3.5 GiB of global
headroom when topping up. Temporary allocations by the other processes can
consume that headroom. Explicit `empty_cache()` calls inside the actor and
rollout are suppressed; retaining the warmed rollout cache keeps its next
inference from needing to reallocate space taken by the actor. Cosmos and
environment processes retain their normal cleanup behavior. Automatic
allocator garbage collection is disabled and expandable native segments are
required. PyTorch can still release cached segments during internal OOM
recovery, and this does not enforce exclusive device access against other
users. Reliable device exclusivity additionally requires scheduler/cgroup
controls. No other jobs are signalled or modified.

With `SPATIAL_WOVR_OPENVLA_WAN_ACTOR=1`, the Cosmos, policy rollout and
environment workers preallocate their own reusable CUDA caches before the
actor sizes its Wan/training pool. Their per-device PyTorch allocator limits
are 11, 13 and 2 GiB respectively. The environment uses two persistent
13-frame buffers and scores one eight-frame action chunk at a time. The actor
derives one limit per GPU from the measured task usage and free memory. It
accounts for the other allocators' remaining capacity, the 3.5 GiB device
headroom and 2.5 GiB for subsequent driver/NCCL allocations. The
`shared_wan_pool` directory records each worker's reservation, the actor's
limit and the memory observed at model and optimizer transitions.

The formal entry retains the optimized 128-environment, 8-rollout protocol,
micro batch 32, accumulation 8, and 20 outer steps; save/evaluate every 5 steps
and full continuation state at 10/20. Pass the ordinary `runner.resume_dir=`
override for a full checkpoint. The probe instead runs two shorter outer
steps, each with eight micro-batch backwards and one optimizer update per
rank, so it exercises both phase transitions and reloading Wan after an
update. It performs no checkpointing or 500-episode evaluation. Only the most
recent update's actor peak report is retained; the device monitor and pool
phase logs span both updates.

Both entries refuse to start GPU workers when a selected card uses over
1024 MiB, rather than waiting or restarting existing work. They do not stop
or replace an active training run. Each private Ray driver also defaults its
Dashboard Agent to a kernel-assigned port, avoiding collisions on Ray's fixed
52365 default when another cluster is running on the same host. Explicitly
configured agent ports are preserved. This is a process-local compatibility
hook; the installed Ray package and other clusters are not modified.
At the time this path was added, CPU
service/trajectory consistency tests and allocator budget tests had passed;
the real two-update GPU probe had not been executed because the existing
eight-GPU run was still active. Run the probe after those GPUs are released:

```bash
screen -dmS spatial-shared-pool-probe \
  bash experiments/run_spatial_wovr_shared_pool_probe_8gpu.sh
```

`<output>/shared_wan_pool/rank-N.jsonl` records actor PID, phase, allocated
tensors, reserved cache, task target, global free memory and reservation errors.
This directory links to the launch's runtime directory. The existing
`gpu_peak/device/` monitor records whole-card consumption during the probe.
Before formal use, verify both updates complete, the recorded Wan owner PID
matches the actor PID, and reserved memory does not collapse during
`wan_offload`, `before_training`, `after_training` and `after_weight_sync`.
