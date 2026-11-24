#!/usr/bin/env bash
# Resume the optimized eight-GPU run after all eight cards become idle.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/.." && pwd)"
stamp="$(date -u +%Y%m%d_%H%M%S)_$$"
default_checkpoint="/data41/qianb2608/dyk/grpo_runs/spatial_wovr20_fast_pi05_12000_8gpu/20261020_131300_node146/spatial_wovr20_wrist_v3_pi05_12000_8gpu_kir_first_close_pre16/checkpoints/global_step_10"
resume_checkpoint="${SPATIAL_WOVR_RESUME_FROM:-$default_checkpoint}"
resume_checkpoint="${resume_checkpoint%/}"
[[ "${resume_checkpoint##*/}" == global_step_10 ]] || { echo "Expected global_step_10 resume directory" >&2; exit 1; }
[[ -s "$resume_checkpoint/actor/model_state_dict/full_weights.pt" && -s "$resume_checkpoint/actor/dcp_checkpoint/.metadata" ]] || { echo "Missing full resume checkpoint: $resume_checkpoint" >&2; exit 1; }
for rank in {0..7}; do
  [[ -s "$resume_checkpoint/actor/dcp_checkpoint/__${rank}_0.distcp" ]] || { echo "Missing optimizer shard for rank $rank" >&2; exit 1; }
done
export SPATIAL_WOVR_OUTPUT="${SPATIAL_WOVR_OUTPUT:-$project_root/../grpo_runs/spatial_wovr20_fast_resume10_8gpu/$stamp}"
export SPATIAL_WOVR_RUNTIME="${SPATIAL_WOVR_RUNTIME:-$project_root/runtime_artifacts/spatial_wovr_fast_8gpu}"
export SPATIAL_WOVR_TMP_ROOT="${SPATIAL_WOVR_TMP_ROOT:-$project_root/../r}"
export SPATIAL_WOVR_GPUS=0,1,2,3,4,5,6,7 SPATIAL_WOVR_PLACEMENT=0-7
export SPATIAL_WOVR_FAST_TRANSFER=1 SPATIAL_WOVR_MONITOR_MEMORY=1
watch_session="${SPATIAL_WOVR_WATCH_SESSION:-spatial-watchdog-resume10-8}"
launch_args=(
  "runner.resume_dir=$resume_checkpoint"
  env.train.cosmos_batch_size=4
  env.train.cosmos_wrist_parallel_generation=false
  env.train.fast_frame_transfer=true
  env.train.skip_successful_env_generation=true
  env.train.video_cfg.record_every_n_episodes=4
  "$@"
)
for arg in "$@"; do
  if [[ "$arg" == --check || "$arg" == --check-assets ]]; then
    exec bash "$script_dir/run_spatial_wovr_20step.sh" "${launch_args[@]}"
  fi
done
[[ ! -e "$SPATIAL_WOVR_OUTPUT/train.log" && ! -e "$SPATIAL_WOVR_OUTPUT/exit_code" ]] || { echo "Use a new output directory: $SPATIAL_WOVR_OUTPUT" >&2; exit 1; }
command -v screen >/dev/null
command -v flock >/dev/null
if screen -ls 2>/dev/null | grep -Fq ".$watch_session"; then
  echo "Watchdog screen already exists: $watch_session" >&2
  exit 1
fi
mkdir -p "$project_root/runtime_artifacts/training_queue"
exec 9>"$project_root/runtime_artifacts/training_queue/resume10_8gpu.lock"
flock -n 9 || { echo "An eight-GPU step-10 resume is already queued/running" >&2; exit 1; }
echo "Resume checkpoint: $resume_checkpoint"
echo "New output: $SPATIAL_WOVR_OUTPUT"
python3 -u - <<'PY'
import datetime
import subprocess
import time

stable = 0
while stable < 3:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=15,
        )
        usage = {int(row.split(",")[0]): float(row.split(",")[1]) for row in result.stdout.splitlines() if row.strip()}
        idle = all(index in usage and usage[index] <= 1024 for index in range(8))
        stable = stable + 1 if idle else 0
        now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        print(f"{now} WAITING_FOR_8_IDLE_GPUS: used_mib={usage}; idle_checks={stable}/3", flush=True)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        stable = 0
        print(f"GPU status unavailable ({type(error).__name__}); waiting", flush=True)
    if stable < 3:
        time.sleep(20)
print("GPU gate passed; resuming from step 10 to total step 20", flush=True)
PY
launcher_pid=$$
(
  exec 9>&-
  while [[ ! -f "$SPATIAL_WOVR_OUTPUT/train.log" || ! -f "$SPATIAL_WOVR_OUTPUT/resolved_config.yaml" ]]; do
    [[ ! -f "$SPATIAL_WOVR_OUTPUT/exit_code" ]] || { echo "Training exited before watchdog initialization" >&2; exit 1; }
    kill -0 "$launcher_pid" 2>/dev/null || { echo "Launcher exited before watchdog initialization" >&2; exit 1; }
    sleep 5
  done
  screen -dmS "$watch_session" env "SPATIAL_WATCH_RUN=$SPATIAL_WOVR_OUTPUT" \
    bash "$script_dir/run_spatial_wovr_watchdog.sh"
  echo "Watchdog screen requested: $watch_session; run=$SPATIAL_WOVR_OUTPUT"
) &
exec bash "$script_dir/run_spatial_wovr_20step.sh" "${launch_args[@]}"
