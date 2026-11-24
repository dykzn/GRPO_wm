#!/usr/bin/env bash
# Collect real truth, update the paired WMs, then resume the latest policy.
set -euo pipefail
bundle_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cycle_dry_run=0
if [[ "${1:-}" == "--dry-run" ]]; then cycle_dry_run=1; shift; fi
cycle_output="${1:-$bundle_root/runs/cycle60}"
if [[ "$cycle_output" != /* ]]; then cycle_output="$PWD/$cycle_output"; fi
cycle_gpus="${WM_CYCLE_GPUS:-0,1,2,3,4,5,6,7}"
cycle_policy_step="${WM_CYCLE_POLICY_STEP:-60}"
cycle_target_step="${WM_CYCLE_TARGET_STEP:-80}"
cycle_wm_steps="${WM_CYCLE_WM_STEPS:-2000}"
cycle_seed="${WM_CYCLE_SEED:-20261021}"
if [[ "$cycle_output" == *$'\n'* ]]; then exit 2; fi
if [[ "$cycle_dry_run" == 0 ]]; then
  if [[ -e "$cycle_output" ]]; then
    printf 'Choose a new cycle output directory: %s\n' "$cycle_output" >&2
    exit 1
  fi
  mkdir -p -- "$cycle_output"
fi
run_cycle_stage() {
  if [[ "$cycle_dry_run" == 1 ]]; then
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}
run_cycle_stage python3 "$bundle_root/reproduction/world_model.py" check \
  --check-runtime --output "$cycle_output/check"
run_cycle_stage python3 "$bundle_root/reproduction/world_model.py" collect \
  --gpus "$cycle_gpus" --per-task 50 --mode train \
  --policy-config "$bundle_root/config/spatial.reference.yaml" \
  --policy-checkpoint "$bundle_root/weights/policy/global_step_$cycle_policy_step" \
  --seed "$cycle_seed" --max-steps 240 --settle-steps 10 \
  --reserve-memory --memory-headroom-gib 3.5 --output "$cycle_output/collection"
run_cycle_stage python3 "$bundle_root/reproduction/world_model.py" prepare \
  --collection "$cycle_output/collection" --replay-root "$bundle_root/data/kir_sources" \
  --val-fraction 0.2 --seed "$cycle_seed" --output "$cycle_output/manifest.json"
run_cycle_stage python3 "$bundle_root/reproduction/world_model.py" inspect \
  --manifest "$cycle_output/manifest.json" --verify-files
run_cycle_stage python3 "$bundle_root/reproduction/world_model.py" train \
  --model both --gpus "$cycle_gpus" --ranks-per-model 4 --accumulation 4 \
  --steps "$cycle_wm_steps" --save-interval 1000 --lr 1e-6 --seed "$cycle_seed" \
  --cosmos-recipe v3-reconstructed \
  --settings "$bundle_root/project/experiments/config/spatial_wm_update.json" \
  --manifest "$cycle_output/manifest.json" --reserve-memory --memory-headroom-gib 3.5 \
  --output "$cycle_output/train"
run_cycle_stage python3 "$bundle_root/reproduction/world_model.py" package \
  --selection "$cycle_output/train/selected_world_models.json" \
  --output "$cycle_output/model_pair"
run_cycle_stage env SPATIAL_WOVR_GPUS="$cycle_gpus" python3 "$bundle_root/reproduction/portable.py" train \
  --resume "$cycle_policy_step" --target "$cycle_target_step" \
  --model-pair "$cycle_output/model_pair/pair.json" --output "$cycle_output/grpo"
