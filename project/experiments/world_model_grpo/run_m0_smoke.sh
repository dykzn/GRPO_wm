#!/usr/bin/env bash
# M0 two-stream smoke launcher.
#
# Overrides via environment variables:
#   GPU=4 NUM_VARIANTS=8 TASK_ID=0 TRIAL_ID=0 LOSS_FORM=mse \
#   REWARD=$WAN_CKPT/taskemb_resnet_rm_retrained_20plus100epoch.pth \
#   ./experiments/world_model_grpo/run_m0_smoke.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../../scripts/env.sh"
PYTHON="${PYTHON:-$RLINF_ENV/bin/python}"
COSMOS_ROOT="$COSMOS_POLICY_ROOT"

GPU="${GPU:-0}"
WAN_CKPT="${WAN_CKPT:-$CHECKPOINT_ROOT/wovr/RLinf-Wan-LIBERO-Goal}"
PI05="${PI05:-$CHECKPOINT_ROOT/pi05/RLinf-Pi05-LIBERO-SFT}"
OUT="${OUT:-$RESULTS_ROOT/world_model_grpo/m0_smoke}"
REWARD="${REWARD:-$WAN_CKPT/taskemb_resnet_rm.pth}"

mkdir -p "$OUT"

cd "$OUT"

CUDA_VISIBLE_DEVICES="$GPU" \
PYTHONPATH="$LIBERO_ROOT:$RLINF_ROOT:$WORLDPILOT_ROOT:$COSMOS_ROOT:$PROJECT_ROOT" \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
MUJOCO_EGL_DEVICE_ID=0 \
LIBERO_ROOT="$LIBERO_ROOT" \
LIBERO_CONFIG_PATH="$LIBERO_ROOT/libero/libero" \
ROBOT_PLATFORM=LIBERO \
LIBERO_TYPE=standard \
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
"$PYTHON" -m experiments.world_model_grpo.m0_two_stream_smoke \
  --config "$RLINF_ROOT/evaluations/libero/wovr_primary_wan_pi05_eval.yaml" \
  --pi05-path "$PI05" \
  --wan-model-path "$WAN_CKPT/model-00001.safetensors" \
  --vae-path "$WAN_CKPT/Wan2.2_VAE.pth" \
  --goal-reward-path "$REWARD" \
  --output-dir "$OUT" \
  --task-suite libero_goal \
  --task-id "${TASK_ID:-0}" \
  --trial-id "${TRIAL_ID:-0}" \
  --num-variants "${NUM_VARIANTS:-8}" \
  --loss-form "${LOSS_FORM:-mse}" \
  --seed "${SEED:-20260823}" \
  "$@"
