#!/usr/bin/env bash
# PPO-first real LIBERO rollout, then synthetic-variant diagnostic.
#
# Example:
#   GPU=6 TASK_ID=7 TRIAL_ID=9 NUM_REAL_EPISODES=4 \
#   OUT=experiments/world_model_grpo/results/m0_ppo_then_synthetic_task7 \
#   ./experiments/world_model_grpo/run_m0_ppo_then_synthetic.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../../scripts/env.sh"
PYTHON="${PYTHON:-$RLINF_ENV/bin/python}"
COSMOS_ROOT="$COSMOS_POLICY_ROOT"

GPU="${GPU:-0}"
# egl_docker maps the selected physical GPU to device 0 inside the
# container.  Preserve an explicitly supplied CUDA_VISIBLE_DEVICES there;
# when running the launcher directly, fall back to the requested GPU.
VISIBLE_GPU="${CUDA_VISIBLE_DEVICES:-$GPU}"
WAN_CKPT="${WAN_CKPT:-$CHECKPOINT_ROOT/wovr/RLinf-Wan-LIBERO-Goal}"
PI05="${PI05:-$RESULTS_ROOT/05_GRPO/goal_grpo_task039_resume_step5_signed_gripperfix_gpu5_r3_20260818/20260818_000713/goal_grpo_task039_resume_step5_signed_gripperfix_gpu5_r3/checkpoints/global_step_20}"
OUT="${OUT:-$RESULTS_ROOT/world_model_grpo/m0_ppo_then_synthetic}"
REWARD="${REWARD:-$WAN_CKPT/taskemb_resnet_rm.pth}"
CONFIG="${CONFIG:-$RLINF_ROOT/evaluations/libero/libero_goal_openpi_pi05_grpo_step25_eval.yaml}"

mkdir -p "$OUT"
cd "$OUT"

CUDA_VISIBLE_DEVICES="$VISIBLE_GPU" \
PYTHONPATH="$LIBERO_ROOT:$RLINF_ROOT:$WORLDPILOT_ROOT:$COSMOS_ROOT:$PROJECT_ROOT" \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
MUJOCO_EGL_DEVICE_ID=0 \
LIBERO_ROOT="$LIBERO_ROOT" \
LIBERO_CONFIG_PATH="$LIBERO_ROOT/libero/libero" \
ROBOT_PLATFORM=LIBERO \
LIBERO_TYPE=standard \
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
"$PYTHON" -m experiments.world_model_grpo.m0_ppo_then_synthetic \
  --config "$CONFIG" \
  --pi05-path "$PI05" \
  --wan-model-path "$WAN_CKPT/model-00001.safetensors" \
  --vae-path "$WAN_CKPT/Wan2.2_VAE.pth" \
  --goal-reward-path "$REWARD" \
  --output-dir "$OUT" \
  --task-suite libero_goal \
  --task-id "${TASK_ID:-7}" \
  --trial-id "${TRIAL_ID:-9}" \
  --num-real-episodes "${NUM_REAL_EPISODES:-4}" \
  --max-decisions "${MAX_DECISIONS:-40}" \
  --num-variants "${NUM_VARIANTS:-8}" \
  --seed "${SEED:-20260824}" \
  "$@"
