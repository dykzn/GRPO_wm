#!/usr/bin/env bash

# Evaluate the two available goal-GRPO checkpoints in isolated EGL containers.
# Host GPU4 runs step30 and host GPU5 runs step40 by default.

set -euo pipefail

script_dir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
# shellcheck disable=SC1091
source "$script_dir/../scripts/env.sh"

project_root="$PROJECT_ROOT"
rlinf_root="$RLINF_ROOT"
python_env="$RLINF_ENV"
libero_root="$LIBERO_ROOT"
egl_root="$EGL_ROOT"
checkpoint_parent="${GOAL_EVAL_CHECKPOINT_PARENT:-$RESULTS_ROOT/05_GRPO/goal_grpo_task039_samewm_continue20_gpu7_20260824_retry6_wrist5000/goal_grpo_task039_samewm_continue_to40_wrist5000_already_signed/checkpoints}"
norm_stats_root="${GOAL_EVAL_NORM_STATS_ROOT:-$CHECKPOINT_ROOT/pi05/RLinf-Pi05-LIBERO-SFT/physical-intelligence/libero}"
step="${GOAL_EVAL_STEP:-30}"
host_gpu="${GOAL_EVAL_HOST_GPU:-4}"
container_gpu=0
result_root="${GOAL_EVAL_RESULT_ROOT:-$project_root/results/04_EVALUATION/goal_grpo_task039_samewm_continue20_wrist5000_step${step}_gpu${host_gpu}_20260904_docker_egl}"
experiment_name="${GOAL_EVAL_EXPERIMENT_NAME:-goal_grpo_task039_samewm_continue20_wrist5000_step${step}_gpu${host_gpu}_8env_50trials}"
screen_name="${GOAL_EVAL_SCREEN:-goal-eval-t039-s${step}-gpu${host_gpu}-egl}"
script_path="$(readlink -f "${BASH_SOURCE[0]}")"
checkpoint="$checkpoint_parent/global_step_${step}/actor"

validate_settings() {
    case "$step" in
        30|40) ;;
        *) echo "Only step30 and step40 checkpoints are available; got step=$step" >&2; return 2 ;;
    esac
    [[ -x "$python_env/bin/python" ]] || {
        echo "Missing Python environment: $python_env/bin/python" >&2
        return 2
    }
    [[ -f "$checkpoint/model_state_dict/full_weights.pt" ]] || {
        echo "Missing checkpoint weights: $checkpoint/model_state_dict/full_weights.pt" >&2
        return 2
    }
    [[ -f "$norm_stats_root/norm_stats.json" ]] || {
        echo "Missing norm stats: $norm_stats_root/norm_stats.json" >&2
        return 2
    }
    [[ -x "$egl_root/run_with_docker_egl.sh" ]] || {
        echo "Missing EGL wrapper: $egl_root/run_with_docker_egl.sh" >&2
        return 2
    }
}

run_inside_container() {
    mkdir -p "$result_root"
    exec > >(tee -a "$result_root/eval.log") 2>&1

    cd "$project_root"
    source "$python_env/bin/activate"

    export RLINF_ROOT="$rlinf_root"
    export REPO_PATH="$rlinf_root"
    export EMBODIED_PATH="$rlinf_root/examples/embodiment"
    export LIBERO_ROOT="$libero_root"
    export LIBERO_CONFIG_PATH="$libero_root/libero/libero"
    export PYTHONPATH="$libero_root:$rlinf_root:${PYTHONPATH:-}"
    export ROBOT_PLATFORM=LIBERO
    export LIBERO_TYPE=standard
    export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
    export CUDA_VISIBLE_DEVICES="$container_gpu"
    export MUJOCO_GL=egl
    export PYOPENGL_PLATFORM=egl
    export MUJOCO_EGL_DEVICE_ID="$container_gpu"
    export NVIDIA_DRIVER_CAPABILITIES=compute,utility
    export JAX_PLATFORMS=cpu
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HYDRA_FULL_ERROR=1
    export RAY_memory_monitor_refresh_ms=0
    export PYTHONUNBUFFERED=1
    export TOKENIZERS_PARALLELISM=false
    export MPLCONFIGDIR="/tmp/goal-eval-t039-s${step}-gpu${host_gpu}-matplotlib"
    unset RAY_ADDRESS
    mkdir -p "$MPLCONFIGDIR"

    printf 'RUNNING\n' > "$result_root/status"
    echo "[$(date '+%F %T')] Starting LIBERO-Goal evaluation inside EGL Docker."
    echo "[$(date '+%F %T')] Host GPU=${host_gpu}; container GPU=${container_gpu}; checkpoint step=${step}."
    echo "[$(date '+%F %T')] Parallel envs=8; trials per task=50; max steps=17520; signed gripper=already_signed."
    echo "[$(date '+%F %T')] Checkpoint=$checkpoint"
    echo "[$(date '+%F %T')] Result root=$result_root"

    set +e
    "$python_env/bin/python" "$rlinf_root/evaluations/eval_embodied_agent.py" \
        --config-path "$rlinf_root/evaluations/libero/" \
        --config-name libero_goal_openpi_pi05_grpo_step25_eval \
        "runner.logger.log_path=$result_root" \
        "runner.logger.experiment_name=$experiment_name" \
        "rollout.model.model_path=$checkpoint" \
        "rollout.model.openpi_data.norm_stats_path=$norm_stats_root" \
        "rollout.model.openpi_data.output_gripper_action_mode=already_signed" \
        env.eval.total_num_envs=8 \
        env.eval.rollout_epoch=1 \
        env.eval.auto_reset=true \
        env.eval.ignore_terminations=true \
        env.eval.max_episode_steps=280 \
        env.eval.max_steps_per_rollout_epoch=17520 \
        env.eval.use_fixed_reset_state_ids=true \
        env.eval.use_ordered_reset_state_ids=true \
        env.eval.is_eval=true \
        env.eval.openpi_gripper_action_mode=already_signed \
        env.eval.video_cfg.save_video=true \
        env.eval.video_cfg.split_by_episode=true \
        +env.eval.video_cfg.stop_video_on_success=false \
        +env.eval.eval_trials_per_task=50
    eval_rc=$?
    set -e

    if (( eval_rc == 0 )); then
        printf 'DONE\n' > "$result_root/status"
        echo "[$(date '+%F %T')] Evaluation finished normally."
    else
        printf 'FAILED\n' > "$result_root/status"
        echo "[$(date '+%F %T')] Evaluation failed with exit code $eval_rc."
    fi
    exit "$eval_rc"
}

case "${1:-}" in
    --inside)
        validate_settings
        run_inside_container
        ;;
    --docker)
        validate_settings
        cd "$project_root"
        export EGL_GPU="$host_gpu"
        export EGL_DOCKER_NETWORK=host
        export EGL_SHM_SIZE="${GOAL_EVAL_EGL_SHM_SIZE:-32g}"
        exec "$egl_root/run_with_docker_egl.sh" --gpu "$host_gpu" -- \
            bash -lc "export GOAL_EVAL_STEP='$step' GOAL_EVAL_HOST_GPU='$host_gpu' GOAL_EVAL_RESULT_ROOT='$result_root' GOAL_EVAL_EXPERIMENT_NAME='$experiment_name' GOAL_EVAL_SCREEN='$screen_name'; exec bash '$script_path' --inside"
        ;;
    "")
        validate_settings
        if screen -ls 2>/dev/null | grep -q "\\.${screen_name}[[:space:]]"; then
            echo "Screen session already exists: $screen_name" >&2
            exit 1
        fi
        if ! command -v docker >/dev/null 2>&1; then
            echo "docker is not available" >&2
            exit 1
        fi
        if ! docker image inspect egl-docker:550.163.01 >/dev/null 2>&1; then
            echo "Missing Docker image: egl-docker:550.163.01" >&2
            exit 1
        fi
        used_mb="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$host_gpu" 2>/dev/null | tr -d '[:space:]')"
        if [[ -z "$used_mb" || "$used_mb" -gt 4000 ]]; then
            echo "Host GPU${host_gpu} is not free (memory.used=${used_mb:-unknown} MiB); refusing to launch." >&2
            exit 1
        fi
        mkdir -p "$result_root"
        printf 'STARTING\n' > "$result_root/status"
        screen -dmS "$screen_name" bash -lc "exec bash '$script_path' --docker >'$result_root/coordinator.log' 2>&1"
        echo "Started $screen_name"
        echo "GPU: $host_gpu"
        echo "Checkpoint: step$step"
        echo "Result root: $result_root"
        echo "Log: $result_root/eval.log"
        ;;
    *)
        echo "Usage: $script_path [--inside|--docker]" >&2
        exit 2
        ;;
esac
