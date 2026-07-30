#!/usr/bin/env bash
set -euo pipefail

export VERL_RUN_ID=${VERL_RUN_ID:-flexexternal3}
export MLFLOW_PROJECT=${MLFLOW_PROJECT:-dynamic-flexkv-sleep-smoke}
export MLFLOW_PORT=${MLFLOW_PORT:-15100}
export PLANNED_TRAINER_STEPS=${PLANNED_TRAINER_STEPS:-1}
export TOTAL_ROLLOUT_STEPS=${TOTAL_ROLLOUT_STEPS:-16}
export EXPECTED_MLFLOW_TRACES=${EXPECTED_MLFLOW_TRACES:-32}
export MINIMUM_RETRY_ATTEMPTS=${MINIMUM_RETRY_ATTEMPTS:-1}
export DYNAMIC_DEACTIVATE_RATIO=${DYNAMIC_DEACTIVATE_RATIO:-0.0625}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
export ACTOR_PPO_MAX_TOKEN_LEN=${ACTOR_PPO_MAX_TOKEN_LEN:-4096}
export INFER_PPO_MAX_TOKEN_LEN=${INFER_PPO_MAX_TOKEN_LEN:-6144}
export FLEXKV_CPU_CACHE_GB=${FLEXKV_CPU_CACHE_GB:-16}
export FLEXKV_INSTANCE_NUM=${FLEXKV_INSTANCE_NUM:-4}
export FLEXKV_SERVER_RECV_PORT=${FLEXKV_SERVER_RECV_PORT:-ipc:///tmp/flexkv_server_${VERL_RUN_ID}}
export FLEXKV_SERVER_LAUNCH_MODE=${FLEXKV_SERVER_LAUNCH_MODE:-external}
export FLEXKV_ENABLE_MPS=${FLEXKV_ENABLE_MPS:-1}

ROOT=${ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/05_claude_ws}
WORKSPACE=${WORKSPACE:-${ROOT}/dynamic_flexkv_abort_dev}
FLEXKV_REPO=${FLEXKV_REPO:-${WORKSPACE}/src/FlexKV}
SERVER_PID=""
MEMORY_SAMPLER_PID=""
SERVER_LOG=${WORKSPACE}/logs/external_flexkv_server_${VERL_RUN_ID}_rank${RANK:-0}.log
MEMORY_LOG=${WORKSPACE}/logs/gpu_memory_${VERL_RUN_ID}_rank${RANK:-0}.log
SERVER_RECV_PATH=${FLEXKV_SERVER_RECV_PORT#ipc://}
GPU_REGISTER_PORT=${FLEXKV_SERVER_RECV_PORT}_gpu_register
GPU_REGISTER_PATH=${GPU_REGISTER_PORT#ipc://}

snapshot_gpu_memory() {
  local phase=$1
  {
    echo "=== $(date --iso-8601=ns) phase=${phase} rank=${RANK:-0} ==="
    nvidia-smi --query-gpu=index,uuid,memory.used,memory.free \
      --format=csv,noheader
    echo "--- compute-apps ---"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name \
      --format=csv,noheader 2>/dev/null || true
  } >>"${MEMORY_LOG}"
}

start_memory_sampler() {
  : >"${MEMORY_LOG}"
  (
    while true; do
      snapshot_gpu_memory periodic
      sleep "${GPU_MEMORY_SAMPLE_INTERVAL_S:-5}"
    done
  ) &
  MEMORY_SAMPLER_PID=$!
}

stop_memory_sampler() {
  if [[ -n "${MEMORY_SAMPLER_PID}" ]]; then
    kill "${MEMORY_SAMPLER_PID}" 2>/dev/null || true
    wait "${MEMORY_SAMPLER_PID}" 2>/dev/null || true
    MEMORY_SAMPLER_PID=""
  fi
}

stop_mps() {
  printf '%s\n' quit | timeout 5s nvidia-cuda-mps-control 2>/dev/null || true
  if pgrep -f '^nvidia-cuda-mps-(server|control)' >/dev/null; then
    pkill -TERM -f '^nvidia-cuda-mps-(server|control)' || true
    sleep 2
  fi
  if pgrep -f '^nvidia-cuda-mps-(server|control)' >/dev/null; then
    pkill -KILL -f '^nvidia-cuda-mps-(server|control)' || true
    sleep 1
  fi
  if pgrep -f '^nvidia-cuda-mps-(server|control)' >/dev/null; then
    echo "failed to stop stale CUDA MPS daemon" >&2
    return 1
  fi
}

stop_external_server() {
  snapshot_gpu_memory shutdown_begin
  stop_memory_sampler
  if [[ -n "${SERVER_PID}" ]]; then
    kill -TERM -- "-${SERVER_PID}" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 -- "-${SERVER_PID}" 2>/dev/null || break
      sleep 0.5
    done
    kill -KILL -- "-${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
    SERVER_PID=""
  fi
  stop_mps || true
  rm -f "${SERVER_RECV_PATH}" "${GPU_REGISTER_PATH}"
  snapshot_gpu_memory shutdown_complete
}
trap stop_external_server EXIT

start_memory_sampler
snapshot_gpu_memory baseline_clean

  stop_mps
  rm -f "${SERVER_RECV_PATH}" "${GPU_REGISTER_PATH}"
  PYTHONPATH="${FLEXKV_REPO}:${PYTHONPATH:-}" \
    setsid python3 "${WORKSPACE}/scripts/run_external_flexkv_node_server.py" \
      --server-recv-port "${FLEXKV_SERVER_RECV_PORT}" \
      --gpu-register-port "${GPU_REGISTER_PORT}" \
      --expected-gpus 8 \
      --instance-num "${FLEXKV_INSTANCE_NUM}" \
      --tp-size 2 \
      --num-layers 48 \
      --num-kv-heads 4 \
      --head-size 256 \
      --tokens-per-block 16 \
      --num-cpu-blocks 10922 \
      --packed-kv \
      --start-mps \
      >"${SERVER_LOG}" 2>&1 &
  SERVER_PID=$!

  for _ in $(seq 1 120); do
    [[ -S "${SERVER_RECV_PATH}" ]] && break
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      cat "${SERVER_LOG}"
      exit 1
    fi
    sleep 1
  done
  test -S "${SERVER_RECV_PATH}"
  snapshot_gpu_memory flexkv_server_ready
  echo "external_flexkv_server_pid=${SERVER_PID}"
  echo "external_flexkv_server_log=${SERVER_LOG}"

snapshot_gpu_memory before_workload
bash "${WORKSPACE}/scripts/dfw_h100_dynamic_recompute_30b_r8k_req32_flexkv.sh"
