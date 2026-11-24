#!/usr/bin/env bash

set -euo pipefail

script_dir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
# shellcheck disable=SC1091
source "$script_dir/../scripts/env.sh"

project_root="$PROJECT_ROOT"
rlinf_root="$RLINF_ROOT"
libero_root="$LIBERO_ROOT"
python_env="$RLINF_ENV"
egl_root="$EGL_ROOT"
checkpoint_root="${PI05_CHECKPOINT_ROOT:-$CHECKPOINT_ROOT/pi05_libero_finetuned_v044}"
wovr_root="${WOVR_GOAL_WAN_ROOT:-$CHECKPOINT_ROOT/wovr/RLinf-Wan-LIBERO-Goal}"
result_base_root="${WOVR_GOAL_RESULT_BASE_ROOT:-$RESULTS_ROOT/wovr_pi05_grpo_libero_goal_state_bridge_egl_gpu1}"
export EGL_EXTRA_MOUNT="${EGL_EXTRA_MOUNT:-$DATA_ROOT}"

config_name="wan_libero_goal_grpo_openpi_pi05_hybrid"
experiment_name="${WOVR_GOAL_EXPERIMENT_NAME:-wovr_pi05_grpo_libero_goal_state_bridge}"
host_gpu="${WOVR_GOAL_HOST_GPU:-1}"
screen_name="${WOVR_GOAL_SCREEN_NAME:-wovr-pi05-grpo-libero-goal-state-bridge-gpu${host_gpu}}"
ray_port="${WOVR_GOAL_RAY_PORT:-29528}"
max_steps="${WOVR_GOAL_MAX_STEPS:-20}"
save_interval="${WOVR_GOAL_SAVE_INTERVAL:-5}"
script_path="$(readlink -f "${BASH_SOURCE[0]}")"

run_inside_container() {
    local result_root="${WOVR_GOAL_RESULT_ROOT:?WOVR_GOAL_RESULT_ROOT is required}"
    local log_file="$result_root/train.log"
    local status_file="$result_root/status"

    mkdir -p "$result_root"
    exec > >(tee -a "$log_file") 2>&1
    printf 'CONTAINER_READY\n' > "$status_file"

    cd "$rlinf_root"
    source "$python_env/bin/activate"

    export RLINF_ROOT="$rlinf_root"
    export EMBODIED_PATH="$rlinf_root/examples/embodiment"
    export LIBERO_ROOT="$libero_root"
    export LIBERO_CONFIG_PATH="$libero_root/libero/libero"
    export PYTHONPATH="$libero_root:$rlinf_root:${PYTHONPATH:-}"
    export ROBOT_PLATFORM=LIBERO
    export LIBERO_TYPE=standard
    export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
    export CUDA_VISIBLE_DEVICES=0
    export MUJOCO_GL=egl
    export PYOPENGL_PLATFORM=egl
    export MUJOCO_EGL_DEVICE_ID=0
    export NVIDIA_DRIVER_CAPABILITIES=compute,utility
    export JAX_PLATFORMS=cpu
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export RAY_memory_monitor_refresh_ms=0
    export PYTHONUNBUFFERED=1
    export TOKENIZERS_PARALLELISM=false
    export MPLCONFIGDIR=/tmp/wovr-pi05-grpo-goal-state-bridge-matplotlib
    mkdir -p "$MPLCONFIGDIR"

    for required_file in \
        "$checkpoint_root/model.safetensors" \
        "$checkpoint_root/config.json" \
        "$checkpoint_root/policy_preprocessor.json" \
        "$checkpoint_root/policy_preprocessor_step_2_normalizer_processor.safetensors" \
        "$wovr_root/model-00001.safetensors" \
        "$wovr_root/Wan2.2_VAE.pth" \
        "$wovr_root/taskemb_resnet_rm.pth"; do
        if [[ ! -f "$required_file" ]]; then
            echo "Missing required file: $required_file" >&2
            printf 'FAILED\n' > "$status_file"
            exit 1
        fi
    done

    echo "[$(date '+%F %T')] Starting Goal WoVR Wan + native LeRobot Pi05 GRPO on host GPU${host_gpu}."
    echo "[$(date '+%F %T')] Pi05 checkpoint: $checkpoint_root"
    echo "[$(date '+%F %T')] WoVR checkpoint: $wovr_root"
    echo "[$(date '+%F %T')] Config: $config_name"
    echo "[$(date '+%F %T')] Max training steps: $max_steps"
    echo "[$(date '+%F %T')] Checkpoint interval: $save_interval"
    echo "[$(date '+%F %T')] State source: LIBERO shadow simulator; visual/reward source: Goal WoVR"

    if ! ray start --head --port="$ray_port" --num-gpus=1 --memory=461708984320; then
        printf 'FAILED\n' > "$status_file"
        exit 1
    fi
    if ! ray status; then
        printf 'FAILED\n' > "$status_file"
        exit 1
    fi
    printf 'RAY_READY\n' > "$status_file"

    python examples/embodiment/train_embodied_agent.py \
        --config-name "$config_name" \
        'cluster.component_placement={actor\,env\,rollout:0}' \
        "env.train.wan_wm_hf_ckpt_path=$wovr_root" \
        "actor.model.model_path=$checkpoint_root" \
        "rollout.model.model_path=$checkpoint_root" \
        "runner.logger.log_path=$result_root" \
        "runner.logger.experiment_name=$experiment_name" \
        "runner.max_steps=$max_steps" \
        "runner.save_interval=$save_interval" &
    train_pid=$!

    startup_timeout_seconds=1800
    startup_deadline=$((SECONDS + startup_timeout_seconds))
    ready=0
    while kill -0 "$train_pid" 2>/dev/null; do
        if grep -Eq 'Generating Rollout Epochs:|Rollout Epoch|Starting rollout' "$log_file"; then
            printf 'READY\n' > "$status_file"
            echo "[$(date '+%F %T')] READY: Goal WoVR Pi05 GRPO entered the rollout loop."
            ready=1
            break
        fi
        if (( SECONDS >= startup_deadline )); then
            break
        fi
        sleep 5
    done

    if (( ready == 0 )); then
        if kill -0 "$train_pid" 2>/dev/null; then
            printf 'STARTUP_TIMEOUT\n' > "$status_file"
            echo "[$(date '+%F %T')] Startup check timed out; training is still running."
        else
            printf 'FAILED\n' > "$status_file"
        fi
    fi

    train_rc=0
    wait "$train_pid" || train_rc=$?
    if (( train_rc != 0 )); then
        printf 'FAILED\n' > "$status_file"
        exit "$train_rc"
    fi
    printf 'DONE\n' > "$status_file"
}

case "${1:-}" in
    --inside)
        run_inside_container
        ;;
    --docker)
        cd "$project_root"
        export EGL_GPU="$host_gpu"
        export EGL_DOCKER_NETWORK=host
        export EGL_SHM_SIZE=32g
        exec "$egl_root/run_with_docker_egl.sh" --gpu "$host_gpu" -- \
            bash -lc "export WOVR_GOAL_RESULT_ROOT='$WOVR_GOAL_RESULT_ROOT' WOVR_GOAL_HOST_GPU='$host_gpu' WOVR_GOAL_RAY_PORT='$ray_port' WOVR_GOAL_MAX_STEPS='$max_steps' WOVR_GOAL_SAVE_INTERVAL='$save_interval' WOVR_GOAL_EXPERIMENT_NAME='$experiment_name'; exec bash '$script_path' --inside"
        ;;
    "")
        if screen -ls 2>/dev/null | grep -q "\.${screen_name}[[:space:]]"; then
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

        run_id="$(date '+%Y%m%d_%H%M%S')"
        export WOVR_GOAL_RESULT_ROOT="$result_base_root/$run_id"
        mkdir -p "$WOVR_GOAL_RESULT_ROOT"
        printf 'STARTING\n' > "$WOVR_GOAL_RESULT_ROOT/status"

        screen -dmS "$screen_name" bash -lc "exec bash '$script_path' --docker"
        echo "已提交 $screen_name，等待 Goal WoVR Pi05 GRPO 进入 rollout。"
        echo "GPU: $host_gpu"
        echo "结果目录: $WOVR_GOAL_RESULT_ROOT"
        echo "日志: $WOVR_GOAL_RESULT_ROOT/train.log"

        startup_timeout_seconds=1800
        startup_deadline=$((SECONDS + startup_timeout_seconds))
        while (( SECONDS < startup_deadline )); do
            status=""
            if [[ -f "$WOVR_GOAL_RESULT_ROOT/status" ]]; then
                status="$(tr -d '[:space:]' < "$WOVR_GOAL_RESULT_ROOT/status" 2>/dev/null || true)"
            fi
            case "$status" in
                READY)
                    echo "✅ GPU${host_gpu} Goal WoVR Pi05 GRPO 已进入 rollout。"
                    echo "Screen: $screen_name"
                    echo "Log: $WOVR_GOAL_RESULT_ROOT/train.log"
                    exit 0
                    ;;
                FAILED)
                    echo "❌ GPU${host_gpu} Goal WoVR Pi05 GRPO 启动失败，最近日志：" >&2
                    tail -n 120 "$WOVR_GOAL_RESULT_ROOT/train.log" 2>/dev/null || true
                    exit 1
                    ;;
            esac
            sleep 10
        done
        echo "⚠️ 启动检查超时，检查日志：$WOVR_GOAL_RESULT_ROOT/train.log" >&2
        exit 1
        ;;
    *)
        echo "Usage: $0 [--inside|--docker]" >&2
        exit 2
        ;;
esac
