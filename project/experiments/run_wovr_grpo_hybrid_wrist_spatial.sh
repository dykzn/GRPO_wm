#!/usr/bin/env bash
# WoVR GRPO - LIBERO Spatial suite (Pi05 stateless + v3 primary-ref wrist).
# Usage:  bash experiments/run_wovr_grpo_hybrid_wrist_spatial.sh
# Optional env: HYBRID_GPU, PROJECT_ROOT, HYBRID_RESULT_BASE, HYBRID_COSMOS_PORT,
#               HYBRID_VALUE_PORT, HYBRID_RAY_PORT, HYBRID_STEPS, HYBRID_SAVE_INTER
set -euo pipefail

script_dir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
# shellcheck disable=SC1091
source "$script_dir/../scripts/env.sh"
project_root="$PROJECT_ROOT"

export HYBRID_GRPO_CONFIG="${HYBRID_GRPO_CONFIG:-wan_libero_spatial_grpo_openpi_pi05_hybrid_wovr_wrist}"
export COSMOS_WRIST_CHECKPOINT="${COSMOS_WRIST_CHECKPOINT:-$project_root/results/cosmos_wovr_wrist_v3_spatial/cosmos_policy/cosmos_wovr_wrist_full/libero_cosmos_wovr_wrist_v3_20260803_130848_2279417_b32_ga1_gpu4/checkpoints/iter_000020000}"
export HYBRID_WAN_CKPT="${HYBRID_WAN_CKPT:-$project_root/checkpoints/wovr/RLinf-Wan-LIBERO-Spatial}"
export HYBRID_GPU="${HYBRID_GPU:-0}"
export HYBRID_COSMOS_PORT="${HYBRID_COSMOS_PORT:-8059}"
export HYBRID_VALUE_PORT="${HYBRID_VALUE_PORT:-8060}"
export HYBRID_RAY_PORT="${HYBRID_RAY_PORT:-29601}"
export HYBRID_STEPS="${HYBRID_STEPS:-100}"
export HYBRID_SAVE_INTER="${HYBRID_SAVE_INTER:-10}"
export HYBRID_VALUE_ENABLED="${HYBRID_VALUE_ENABLED:-false}"
export HYBRID_RESULT_BASE="${HYBRID_RESULT_BASE:-$project_root/results/wovr_grpo_hybrid_wrist_spatial_stateless}"
export HYBRID_SCREEN="${HYBRID_SCREEN:-wovr-grpo-hybrid-wrist-spatial-stateless}"

exec bash "$script_dir/run_wovr_grpo_hybrid_wrist.sh"
