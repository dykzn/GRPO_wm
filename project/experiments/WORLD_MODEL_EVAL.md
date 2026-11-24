# Local Spatial world-model evaluation

This evaluates the released Wan Spatial primary-view model and custom
Cosmos-WoVR Spatial wrist v3 step-20000 model. Each model is compared to the
corresponding real camera on exactly the same episode IDs and action chunks.
This is a local diagnostic. The available training records do not establish
that these public trajectories were held out from both models.

The source is the existing local
`/data41/qianb2608/wan/hf_home/LIBERO-Cosmos-Policy/all_episodes` directory.
The evaluator never treats cached Cosmos subgoal images as real observations.
It uses both successful and failed Spatial trajectories.

## Run

Prepare a small manifest and check availability without CUDA inference:

```bash
bash experiments/run_world_model_eval.sh prepare
bash experiments/run_world_model_eval.sh check
```

`check` reports missing assets and visible CUDA devices; exit code 2 means the
job is not ready. It neither downloads weights nor starts training/inference.
The default manifest is `runtime_artifacts/wm_eval/manifest.json`. By default,
inventory reads actions, initial real camera frames and stream lengths. Add
`--verify-frames` with a new manifest path to decode every scored frame up
front; this can take time on NFS. Reads retry transient I/O errors. Explicit
inventory exclusions are recorded. Prediction never silently skips a failed
episode: both models must finish the manifest before scoring.

Install the metric dependencies and public feature/flow weights:

```bash
bash scripts/setup_world_model_metrics.sh
```

This installs LPIPS, pytorch-fid, CuPy and fastrlock into an isolated
`runtime_artifacts/wm_metrics/packages` directory. Existing Python training
environments are not modified. It downloads AlexNet, FID Inception, PWCNet,
and the I3D TorchScript export from upstream repositories/public model releases,
using direct connections without proxies. Large world-model
checkpoints are reused. `asset_manifest.json` records feature-weight hashes
and the FloLPIPS source commit. Installation needs network access in the
user's terminal. Metric execution itself rejects unexpected weight downloads.

The original StyleGAN-V I3D Dropbox endpoint times out on this host. Setup
instead uses the identical 51,235,320-byte file in ByteDance's public
LatentSync release, pinned to a commit. Its SHA-256 must match the copy
published by LanguageBind/Open-Sora-Plan:
`bec6519f66ea534e953026b4ae2c65553c17bf105611c746d904657e5860a5e2`.
Downloaded partials are promoted only after this check. Completed metric
weights are reused. To finish a previous setup that failed only at I3D:

```bash
bash scripts/setup_world_model_metrics.sh --i3d-only
```

`HF_ENDPOINT` selects the Hugging Face host (default `https://huggingface.co`);
for the previously authorized mirror, use
`HF_ENDPOINT=https://hf-mirror.com bash scripts/setup_world_model_metrics.sh --i3d-only`.
`WM_I3D_URL` overrides that one file's URL. Alternatively set
`WM_I3D_LOCAL_FILE=/path/to/i3d_torchscript.pt` to reuse a separately
downloaded copy. Every source must pass the same hash check; its origin is
recorded in `asset_manifest.json`. `--i3d-only` skips pip, Git and the other
three downloads, then validates all four cached networks on CPU.

Run in screen to survive terminal disconnection:

```bash
screen -S spatial-wm-metrics
bash experiments/run_world_model_eval.sh run --gpus 0,1,2,3,4,5,6,7
```

Detach with Ctrl+A then D. Default output is
`/data73/qianb2608/wm_eval/spatial_<UTC timestamp>/`. Each GPU runs an
independent batch-one shard by default; Wan finishes before Cosmos starts. After the
model processes exit, metrics run on the first listed GPU. `metric_smoke.log`
validates the actual metric backbones and CUDA optical-flow kernels before
long generation begins. Scoring processes expose the metric interpreter's
existing PyTorch/NVIDIA pip CUDA libraries through `LD_LIBRARY_PATH` before
importing CuPy; the generation runtimes retain their own library paths.
`wan_shardN.log`, `cosmos_shardN.log`, and
`metrics.log` show progress. Ctrl+C/SIGTERM cleans up only processes created
by this invocation. No Ray head or GRPO learner is started.

To try two independent generation processes per GPU:

```bash
bash experiments/run_world_model_eval.sh run --gpus 0,1,2,3,4,5,6,7 \
  --workers-per-gpu 2 --output /path/to/original/output
```

Stop the previous invocation with Ctrl+C before resuming the same output
directory. Sixteen disjoint episode shards share the eight GPUs; each
process retains batch one and its own model/state. Complete cached videos
are reused after the usual settings checks, even when shard count changes.
Start with two processes per GPU and compare actual throughput; available
VRAM alone does not establish that compute capacity is idle.

New invocations put per-worker logs under `executions/<time>_<pid>/` alongside
the GPU snapshot, slot assignment, and per-model throughput JSON. Throughput
there is newly generated frames divided by the whole model phase wall time,
including loading, warmup, I/O and saving. The per-process FPS in episode
metadata remains a latency measurement under the concurrent load; it is not
an isolated single-process speed benchmark. Quality metrics retain the same
frames, actions and seeds.

Before model replicas load, `cosmos_code_check.log` verifies the actual
temporal inference imports without allocating model weights. The 13-frame
layout constants live in `cosmos_policy.models.temporal_video_layout`, so
standalone inference does not require the omitted temporal training dataset.
Cosmos also generates one uncached real episode on one GPU first, recorded
in `cosmos_first_episode.log`, before launching the remaining replicas.
That episode uses the full original manifest identity and is reused by the
normal shards.

A short end-to-end smoke can use ten episodes, round-robin across tasks:

```bash
bash experiments/run_world_model_eval.sh prepare --limit 10 \
  --manifest runtime_artifacts/wm_eval/smoke10.json
bash experiments/run_world_model_eval.sh run --gpus 0 \
  --manifest runtime_artifacts/wm_eval/smoke10.json
```

To resume a failed run, pass its original `--output` directory and manifest.
Completed episodes are reused only if settings, model files, GPU model,
PyTorch version and code fingerprints match. Source episodes must retain
the size and modification time recorded in the manifest. Reuse requires the
same settings, including `--steps`, `--seed` and `--warmup`.

## Temporal and image protocol

For an episode with N observations, the evaluator uses
`8 * floor(min(N-1, number_of_actions) / 8)` future frames. Action i is paired
with real observation i+1. It supplies only real o0, repeats o0 for the first
memory window, and then rolls generated frames into memory. There is no
teacher forcing, terminal-frame repetition, repeated-action padding, or
reset to real future frames. At most seven available tail steps are unused.
Trajectories shorter than 16 future frames are recorded as exclusions.

Source Cosmos JPEGs are converted from `cosmos_model_flipud` to
`rlinf_runtime_rot180` by the project's horizontal-reflection convention.
Dataset actions are already signed LIBERO commands and are passed through
without another Pi05 gripper conversion. Cosmos normalizes them with its
local dataset statistics; Wan receives its usual raw residual commands.

Wan retains its native 256x256 generation resolution. Cosmos wrist retains
224x224, the primary image in its fixed reference slot, and the v3 wrist
memory update. All quality metrics use 224x224 RGB. Wan predictions and
primary truth are resized; wrist truth receives the same center crop as the
Cosmos wrist input, while generated wrist frames are not cropped twice.
Conditioning frames are excluded from every quality metric.

Both models use five denoising steps by default. Their native resolutions
and camera targets differ; report those columns alongside FPS and quality,
instead of interpreting this as a same-view, same-workload ranking.

## Metrics and output

`metrics.csv` and `metrics.json` contain separate primary/wrist rows:

| Field | Definition |
| --- | --- |
| FPS | Sum of generated frames divided by synchronized prediction wall time; batch one per GPU, no multiplication by GPU count. Includes conditioning/VAE/decoding/CPU output; excludes loading, warmup, disk I/O, metric resize and scoring. |
| LPIPS | AlexNet LPIPS v0.1; mean over all valid predicted frames. |
| FID | pytorch-fid Inception-v3 2048-d features over all predicted/real frames. |
| FVD_variable_length | One TF-Hub-compatible I3D 400-d feature per actual-length video; real/predicted lengths match exactly, with no temporal resizing/padding. This variable-length diagnostic is not the paper's fixed-horizon FVD. |
| FloLPIPS | Official PWCNet/AlexNet FloLPIPS video formula on adjacent predicted/real frame pairs, averaged over valid pairs; no SSIM/other-flow substitute. |

Per-episode metrics and synchronized chunk times are retained. The report
also includes episode-macro LPIPS/FloLPIPS, frame counts, min/max horizons,
GPU/PyTorch/CuDNN details, source manifest identity, and checkpoint identity.
Non-finite metrics fail explicitly. Exact identical adjacent video pairs
have FloLPIPS zero; the upstream formula otherwise divides by zero flow
difference for that identity case. Singular sample covariance is handled
with a symmetric PSD matrix square root in double precision.

`COSMOS_DISABLE_CUDNN=1` retains the existing Cosmos runtime workaround;
the setting is recorded. Override it only if the local GPU/driver runtime
supports the cuDNN path. Check `gpu_preflight.log` for concurrent GPU use when
interpreting throughput. GPU inference and scoring cannot be validated in
the restricted tool process if `/dev/nvidia*` is unavailable.

Metric implementations and feature sources:

- [LPIPS authors' implementation](https://github.com/richzhang/PerceptualSimilarity)
- [pytorch-fid implementation and FID-specific Inception weights](https://github.com/mseitzer/pytorch-fid)
- [Original FVD reference](https://github.com/google-research/google-research/tree/master/frechet_video_distance)
- [StyleGAN-V TF-Hub-compatible I3D export and preprocessing](https://github.com/universome/stylegan-v/blob/master/src/metrics/frechet_video_distance.py)
- [Official FloLPIPS/PWCNet implementation](https://github.com/danier97/flolpips)
