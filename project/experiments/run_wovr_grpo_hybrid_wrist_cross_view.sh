#!/usr/bin/env bash
set -euo pipefail

script_dir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
# shellcheck disable=SC1091
source "$script_dir/../scripts/env.sh"
: "${COSMOS_WRIST_CHECKPOINT:?Set COSMOS_WRIST_CHECKPOINT to a trained v4 cross-view checkpoint}"

export COSMOS_WRIST_CONFIG="cosmos_predict2_2b_480p_libero_cosmos_wovr_wrist_cross_view__inference_only"
export COSMOS_WRIST_EXPECT_CROSS_VIEW=true

exec "${PROJECT_ROOT}/experiments/run_wovr_grpo_hybrid_wrist.sh" "$@"
