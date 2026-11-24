#!/usr/bin/env bash
# Actor-owned Wan with retained CUDA memory; CPU checks do not start workers.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/.." && pwd)"
stamp="$(date -u +%Y%m%d_%H%M%S)_$$"
export SPATIAL_WOVR_OUTPUT="${SPATIAL_WOVR_OUTPUT:-$project_root/grpo_runs/spatial_wovr20_shared_pool_8gpu/$stamp}"
export SPATIAL_WOVR_RUNTIME="${SPATIAL_WOVR_RUNTIME:-$project_root/runtime_artifacts/spatial_wovr_shared_pool_8gpu}"
export SPATIAL_WOVR_TMP_ROOT="${SPATIAL_WOVR_TMP_ROOT:-$project_root/../r}"
export SPATIAL_WOVR_FAST_TRANSFER=1
# Fail before launching Cosmos if any selected card is busy. This does not wait
# for, evict, or signal any existing process. CPU checks bypass the GPU probe.
cpu_check=false
for argument in "$@"; do
  if [[ "$argument" == --check || "$argument" == --check-assets ]]; then cpu_check=true; fi
done
if [[ "$cpu_check" == false ]]; then
  selected_gpus="${SPATIAL_WOVR_GPUS:-0,1,2,3,4,5,6,7}"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
    awk -F, -v selected="$selected_gpus" '
      BEGIN { n=split(selected, ids, ","); for(i=1;i<=n;i++) wanted[ids[i]]=1 }
      { gsub(/ /,"",$1); if ($1 in wanted) { found++; if ($2+0>1024) { print "GPU " $1 " busy (" $2+0 " MiB); no workers launched." > "/dev/stderr"; busy=1 } } }
      END { if(found!=n || busy) exit 1 }'
fi
exec bash "$script_dir/run_spatial_wovr_20step.sh" \
  actor.shared_wan_pool.enabled=true \
  actor.shared_wan_pool.target_job_gib=75.0 \
  actor.shared_wan_pool.device_headroom_gib=3.5 \
  runner.memory_guard.enabled=false \
  env.train.cosmos_batch_size=4 \
  env.train.cosmos_wrist_parallel_generation=false \
  env.train.fast_frame_transfer=true \
  env.train.skip_successful_env_generation=true \
  env.train.video_cfg.record_every_n_episodes=4 \
  runner.logger.experiment_name=spatial_wovr20_shared_pool_wrist_v3_pi05_12000_8gpu \
  "$@"
