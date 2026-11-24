#!/usr/bin/env bash
set -euo pipefail

script_dir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
# shellcheck disable=SC1091
source "$script_dir/../scripts/env.sh"

P="$PROJECT_ROOT"
SUITE="${GRPO_EVAL_SUITE:?GRPO_EVAL_SUITE required: object|goal|spatial|10}"
MODEL_PATH="${GRPO_EVAL_MODEL_PATH:?GRPO_EVAL_MODEL_PATH required}"
GPU_ID="${GRPO_EVAL_GPU:-3}"
EVAL_LABEL="${GRPO_EVAL_LABEL:-grpo_step25}"
RESULT_BASE="${GRPO_EVAL_RESULT_BASE:-$RESULTS_ROOT/wovr_pi05_grpo_${SUITE}_${EVAL_LABEL}_libero_eval}"
SCREEN_NAME="${GRPO_EVAL_SCREEN:-wovr-${EVAL_LABEL}-${SUITE}-libero-eval-gpu${GPU_ID}}"
TOTAL_NUM_ENVS="${GRPO_EVAL_TOTAL_NUM_ENVS:-4}"
ROLLOUT_EPOCH="${GRPO_EVAL_ROLLOUT_EPOCH:-1}"
EVAL_TRIALS_PER_TASK="${GRPO_EVAL_TRIALS_PER_TASK:-}"
MAX_EPISODE_STEPS="${GRPO_EVAL_MAX_EPISODE_STEPS:-280}"
MAX_STEPS_PER_ROLLOUT_EPOCH="${GRPO_EVAL_MAX_STEPS_PER_ROLLOUT_EPOCH:-280}"
SAVE_VIDEO="${GRPO_EVAL_SAVE_VIDEO:-true}"
STOP_VIDEO_ON_SUCCESS="${GRPO_EVAL_STOP_VIDEO_ON_SUCCESS:-false}"
NORM_STATS_PATH="${GRPO_EVAL_NORM_STATS_PATH:-}"
TASK_ID_FILTER="${GRPO_EVAL_TASK_ID_FILTER:-}"
GRIPPER_ACTION_MODE="${GRPO_EVAL_GRIPPER_ACTION_MODE:-}"
# Keep Ray's host-side worker pool bounded.  The evaluator only uses
# TOTAL_NUM_ENVS simulator environments; allowing Ray to detect every logical
# CPU can otherwise pre-start hundreds of idle Python workers.
RAY_NUM_CPUS="${GRPO_EVAL_RAY_NUM_CPUS:-16}"

case "$SUITE" in
    object)  EVAL_CONFIG="libero_object_openpi_pi05_step25_eval" ;;
    goal)    EVAL_CONFIG="libero_goal_openpi_pi05_grpo_step25_eval" ;;
    spatial) EVAL_CONFIG="libero_spatial_openpi_pi05_grpo_step25_eval" ;;
    10)      EVAL_CONFIG="libero_10_openpi_pi05_grpo_step25_eval" ;;
    *) echo "Unknown suite: $SUITE" >&2; exit 2 ;;
esac
if [[ ! -f "$RLINF_ROOT/evaluations/libero/$EVAL_CONFIG.yaml" ]]; then
    echo "Missing eval config: $EVAL_CONFIG" >&2
    exit 2
fi

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"

run_eval_inside() {
    local result_root="${GRPO_EVAL_RESULT_ROOT:?missing result root}"
    local -a eval_trial_override=()
    local -a norm_stats_override=()
    local -a task_filter_override=()
    local -a gripper_action_override=()
    local -a stop_video_override=(
        "++env.eval.video_cfg.stop_video_on_success=$STOP_VIDEO_ON_SUCCESS"
    )
    if [[ -n "$EVAL_TRIALS_PER_TASK" ]]; then
        eval_trial_override=("+env.eval.eval_trials_per_task=$EVAL_TRIALS_PER_TASK")
    fi
    if [[ -n "$NORM_STATS_PATH" ]]; then
        norm_stats_override=("rollout.model.openpi_data.norm_stats_path=$NORM_STATS_PATH")
    fi
    if [[ -n "$TASK_ID_FILTER" ]]; then
        task_filter_override=("+env.eval.task_id_filter=$TASK_ID_FILTER")
    fi
    if [[ -n "$GRIPPER_ACTION_MODE" ]]; then
        # Use Hydra's override-or-add form: experiment configs may already
        # declare the gripper contract explicitly.
        gripper_action_override=("++env.eval.openpi_gripper_action_mode=$GRIPPER_ACTION_MODE")
    fi
    cd "$RLINF_ROOT"
    source "$RLINF_ENV/bin/activate"
    export RLINF_ROOT LIBERO_ROOT LIBERO_CONFIG_PATH ROBOT_PLATFORM=LIBERO LIBERO_TYPE=standard
    export LIBERO_ROOT="$LIBERO_ROOT"
    export LIBERO_CONFIG_PATH="$LIBERO_ROOT/libero/libero"
    export EMBODIED_PATH="$RLINF_ROOT/examples/embodiment"
    export PYTHONPATH="$LIBERO_ROOT:$RLINF_ROOT:${PYTHONPATH:-}"
    export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
    export CUDA_VISIBLE_DEVICES=0
    export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
    export NVIDIA_DRIVER_CAPABILITIES=compute,utility
    export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export RAY_memory_monitor_refresh_ms=0 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
    export RLINF_RAY_NUM_CPUS="$RAY_NUM_CPUS"
    export MPLCONFIGDIR=/tmp/grpo-eval-${EVAL_LABEL}-matplotlib
    mkdir -p "$MPLCONFIGDIR"
    bash evaluations/run_eval.sh libero "$EVAL_CONFIG" \
        "runner.logger.log_path=$result_root" \
        "runner.logger.experiment_name=libero_${SUITE}_openpi_pi05_${EVAL_LABEL}_eval_gpu${GPU_ID}" \
        "rollout.model.model_path=$MODEL_PATH" \
        "env.eval.total_num_envs=$TOTAL_NUM_ENVS" \
        "env.eval.rollout_epoch=$ROLLOUT_EPOCH" \
        "env.eval.max_episode_steps=$MAX_EPISODE_STEPS" \
        "env.eval.max_steps_per_rollout_epoch=$MAX_STEPS_PER_ROLLOUT_EPOCH" \
        "env.eval.use_ordered_reset_state_ids=true" \
        "env.eval.video_cfg.save_video=$SAVE_VIDEO" \
        env.eval.video_cfg.split_by_episode=true \
        "${stop_video_override[@]}" \
        "${norm_stats_override[@]}" \
        "${task_filter_override[@]}" \
        "${gripper_action_override[@]}" \
        "${eval_trial_override[@]}" \
        >"$result_root/eval.log" 2>&1
}

case "${1:-}" in
    --eval-inside) run_eval_inside ;;
    --docker)
        cd "$P"
        export EGL_GPU="$GPU_ID"
        export EGL_DOCKER_NETWORK=host
        export EGL_SHM_SIZE=32g
        export EGL_EXTRA_MOUNT="$DATA_ROOT"
        exec "$EGL_ROOT/run_with_docker_egl.sh" --gpu "$GPU_ID" -- \
            bash -lc "export GRPO_EVAL_RESULT_ROOT='$GRPO_EVAL_RESULT_ROOT'; export GRPO_EVAL_SUITE='$SUITE'; export GRPO_EVAL_MODEL_PATH='$MODEL_PATH'; export GRPO_EVAL_GPU='$GPU_ID'; export GRPO_EVAL_LABEL='$EVAL_LABEL'; export GRPO_EVAL_TOTAL_NUM_ENVS='$TOTAL_NUM_ENVS'; export GRPO_EVAL_ROLLOUT_EPOCH='$ROLLOUT_EPOCH'; export GRPO_EVAL_TRIALS_PER_TASK='$EVAL_TRIALS_PER_TASK'; export GRPO_EVAL_MAX_EPISODE_STEPS='$MAX_EPISODE_STEPS'; export GRPO_EVAL_MAX_STEPS_PER_ROLLOUT_EPOCH='$MAX_STEPS_PER_ROLLOUT_EPOCH'; export GRPO_EVAL_SAVE_VIDEO='$SAVE_VIDEO'; export GRPO_EVAL_STOP_VIDEO_ON_SUCCESS='$STOP_VIDEO_ON_SUCCESS'; export GRPO_EVAL_NORM_STATS_PATH='$NORM_STATS_PATH'; export GRPO_EVAL_TASK_ID_FILTER='$TASK_ID_FILTER'; export GRPO_EVAL_GRIPPER_ACTION_MODE='$GRIPPER_ACTION_MODE'; export GRPO_EVAL_RAY_NUM_CPUS='$RAY_NUM_CPUS'; exec bash '$SCRIPT_PATH' --eval-inside"
        ;;
    "")
        run_id="$(date '+%Y%m%d_%H%M%S')"
        export GRPO_EVAL_RESULT_ROOT="$RESULT_BASE/$run_id"
        mkdir -p "$RESULT_BASE" "$GRPO_EVAL_RESULT_ROOT"
        printf 'STARTING\n' >"$GRPO_EVAL_RESULT_ROOT/status"
        screen -dmS "$SCREEN_NAME" bash -lc \
            "export GRPO_EVAL_RESULT_ROOT='$GRPO_EVAL_RESULT_ROOT' GRPO_EVAL_SUITE='$SUITE' GRPO_EVAL_MODEL_PATH='$MODEL_PATH' GRPO_EVAL_GPU='$GPU_ID' GRPO_EVAL_LABEL='$EVAL_LABEL' GRPO_EVAL_TOTAL_NUM_ENVS='$TOTAL_NUM_ENVS' GRPO_EVAL_ROLLOUT_EPOCH='$ROLLOUT_EPOCH' GRPO_EVAL_TRIALS_PER_TASK='$EVAL_TRIALS_PER_TASK' GRPO_EVAL_MAX_EPISODE_STEPS='$MAX_EPISODE_STEPS' GRPO_EVAL_MAX_STEPS_PER_ROLLOUT_EPOCH='$MAX_STEPS_PER_ROLLOUT_EPOCH' GRPO_EVAL_SAVE_VIDEO='$SAVE_VIDEO' GRPO_EVAL_STOP_VIDEO_ON_SUCCESS='$STOP_VIDEO_ON_SUCCESS' GRPO_EVAL_NORM_STATS_PATH='$NORM_STATS_PATH' GRPO_EVAL_TASK_ID_FILTER='$TASK_ID_FILTER' GRPO_EVAL_GRIPPER_ACTION_MODE='$GRIPPER_ACTION_MODE' GRPO_EVAL_RAY_NUM_CPUS='$RAY_NUM_CPUS'; exec bash '$SCRIPT_PATH' --docker >>'$GRPO_EVAL_RESULT_ROOT/coordinator.log' 2>&1"
        echo "screen=$SCREEN_NAME"
        echo "result_root=$GRPO_EVAL_RESULT_ROOT"
        ;;
    *) echo "Usage: $0" >&2; exit 2 ;;
esac
