#!/usr/bin/env bash

configure_mooncake_abc() {
  KV_ARGS=()
  MOONCAKE_MASTER_PID=""
  if [[ "${ABC_GROUP}" == "A" ]]; then
    export VERL_KV_OFFLOAD_PUT_STATUS=disabled
    return
  fi

  local mooncake_cuda_lib="/usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime/lib"
  if ! command -v mooncake_master >/dev/null 2>&1; then
    local mooncake_wheels=("${WORKSPACE}"/wheelhouse/mooncake_transfer_engine-*.whl)
    [[ -e "${mooncake_wheels[0]}" ]] || { echo "missing cached Mooncake wheel" >&2; return 1; }
    python3 -m pip install --no-deps "${mooncake_wheels[0]}"
  fi
  if [[ ! -s "${mooncake_cuda_lib}/libcudart.so.12" ]]; then
    local cudart_wheels=("${WORKSPACE}"/wheelhouse/nvidia_cuda_runtime_cu12-*.whl)
    [[ -e "${cudart_wheels[0]}" ]] || { echo "missing cached CUDA runtime wheel" >&2; return 1; }
    python3 -m pip install --no-deps "${cudart_wheels[0]}"
  fi
  export LD_LIBRARY_PATH="${mooncake_cuda_lib}:${LD_LIBRARY_PATH:-}"
  python3 -P -c "import mooncake.store"
  command -v mooncake_master >/dev/null

  export VLLM_MOONCAKE_ABORT_KV_REUSE=1
  export VLLM_MOONCAKE_ABORT_KV_TRACE=1
  export MOONCAKE_STRICT_REUSE_TRACE=1
  export VERL_KV_OFFLOAD_PUT_STATUS=enabled
  if [[ "${ABC_GROUP}" == "B" ]]; then
    export VLLM_MOONCAKE_FORCE_MISS=1
  else
    export VLLM_MOONCAKE_FORCE_MISS=0
  fi

  MOONCAKE_MASTER_PORT=${MOONCAKE_MASTER_PORT:-15010}
  MOONCAKE_METRICS_PORT=${MOONCAKE_METRICS_PORT:-15011}
  MOONCAKE_CONFIG_PATH=${MOONCAKE_CONFIG_PATH:-${MLFLOW_RUN_DIR}/mooncake_config.json}
  MOONCAKE_READY_FILE=${LOGDIR}/mooncake_ready_${JOB_ID}.txt
  MOONCAKE_FAIL_FILE=${LOGDIR}/mooncake_failed_${JOB_ID}.txt
  export MOONCAKE_CONFIG_PATH MOONCAKE_MASTER_PORT MOONCAKE_METRICS_PORT

  KV_ARGS=(
    "actor_rollout_ref.rollout.abort_kv_reuse.enabled=True"
    "actor_rollout_ref.rollout.abort_kv_reuse.timeout_s=60"
    "actor_rollout_ref.rollout.kv_backend=mooncake_store"
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_connector=MooncakeStoreConnector"
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_role=kv_both"
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_connector_extra_config.mooncake_config_path=${MOONCAKE_CONFIG_PATH}"
    "+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_MOONCAKE_ABORT_KV_REUSE='1'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_MOONCAKE_ABORT_KV_TRACE='1'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_MOONCAKE_FORCE_MISS='${VLLM_MOONCAKE_FORCE_MISS}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.MOONCAKE_STRICT_REUSE_TRACE='1'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.MOONCAKE_CONFIG_PATH=${MOONCAKE_CONFIG_PATH}"
    "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_KV_OFFLOAD_PUT_STATUS=enabled"
  )
}


setup_mooncake_abc() {
  if [[ "${ABC_GROUP}" == "A" ]]; then
    return
  fi
  export MOONCAKE_REQUESTER_LOCAL_HOSTNAME="${THIS_IP}"
  if [[ "${IS_HEAD}" == "1" ]]; then
    rm -f "${MOONCAKE_READY_FILE}" "${MOONCAKE_FAIL_FILE}"
    {
      cat > "${MOONCAKE_CONFIG_PATH}" <<JSON
{
  "mode": "embedded",
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "${THIS_IP}:${MOONCAKE_MASTER_PORT}",
  "global_segment_size": "32GB",
  "local_buffer_size": "4GB",
  "protocol": "tcp",
  "device_name": "",
  "enable_offload": false
}
JSON
      mooncake_master --port "${MOONCAKE_MASTER_PORT}" --metrics_port "${MOONCAKE_METRICS_PORT}" --default_kv_lease_ttl=600000 \
        > "${LOGDIR}/mooncake_master_${JOB_ID}.log" 2>&1 &
      MOONCAKE_MASTER_PID=$!
      echo "${MOONCAKE_MASTER_PID}" > "${LOGDIR}/mooncake_master_${JOB_ID}.pid"
      python3 -P - <<PY
import socket
import time
host = "${THIS_IP}"
port = int("${MOONCAKE_MASTER_PORT}")
deadline = time.time() + 60
while time.time() < deadline:
    try:
        with socket.create_connection((host, port), timeout=2):
            print(f"mooncake_master_connect=OK {host}:{port}")
            raise SystemExit(0)
    except OSError:
        time.sleep(1)
raise SystemExit("mooncake master did not become reachable")
PY
    } || {
      status=$?
      echo "mooncake setup failed with status ${status}" > "${MOONCAKE_FAIL_FILE}"
      return "${status}"
    }
    echo ready > "${MOONCAKE_READY_FILE}"
  else
    for _ in $(seq 1 "${HEAD_WAIT_SECONDS}"); do
      [[ -s "${MOONCAKE_FAIL_FILE}" || -s "${MOONCAKE_READY_FILE}" ]] && break
      sleep 1
    done
    [[ ! -s "${MOONCAKE_FAIL_FILE}" ]]
    test -s "${MOONCAKE_READY_FILE}"
  fi
  echo "mooncake_group=${ABC_GROUP} requester=${MOONCAKE_REQUESTER_LOCAL_HOSTNAME} config=${MOONCAKE_CONFIG_PATH}"
}


cleanup_mooncake_abc() {
  if [[ "${ABC_GROUP}" == "A" ]]; then
    return
  fi
  if [[ -z "${MOONCAKE_MASTER_PID:-}" && -s "${LOGDIR}/mooncake_master_${JOB_ID}.pid" ]]; then
    MOONCAKE_MASTER_PID=$(cat "${LOGDIR}/mooncake_master_${JOB_ID}.pid" || true)
  fi
  if [[ -n "${MOONCAKE_MASTER_PID:-}" ]]; then
    kill "${MOONCAKE_MASTER_PID}" 2>/dev/null || true
  fi
}
