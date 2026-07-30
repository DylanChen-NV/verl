#!/usr/bin/env bash

configure_flexkv() {
  export VLLM_ABORT_KV_OFFLOAD=${VLLM_ABORT_KV_OFFLOAD:-1}
  export VLLM_CUMEM_ENABLE_SHAREABLE_HANDLE=${VLLM_CUMEM_ENABLE_SHAREABLE_HANDLE:-1}
  export FLEXKV_CPU_CACHE_GB=${FLEXKV_CPU_CACHE_GB:-16}
  export FLEXKV_INSTANCE_NUM=${FLEXKV_INSTANCE_NUM:-1}
  export FLEXKV_SERVER_RECV_PORT=${FLEXKV_SERVER_RECV_PORT:-ipc:///tmp/flexkv_server}
  export FLEXKV_SERVER_LAUNCH_MODE=${FLEXKV_SERVER_LAUNCH_MODE:-embedded}
  export FLEXKV_ENABLE_MPS=${FLEXKV_ENABLE_MPS:-1}
  export VERL_KV_OFFLOAD_PUT_STATUS=enabled

  KV_ARGS=(
    "actor_rollout_ref.rollout.abort_kv_reuse.enabled=True"
    "actor_rollout_ref.rollout.abort_kv_reuse.timeout_s=${ABORT_KV_REUSE_TIMEOUT_S:-120}"
    "actor_rollout_ref.rollout.kv_backend=flexkv"
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_connector=FlexKVConnectorV1"
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_role=kv_both"
    "+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_ABORT_KV_OFFLOAD='${VLLM_ABORT_KV_OFFLOAD}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_CUMEM_ENABLE_SHAREABLE_HANDLE='${VLLM_CUMEM_ENABLE_SHAREABLE_HANDLE}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.FLEXKV_CPU_CACHE_GB='${FLEXKV_CPU_CACHE_GB}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.FLEXKV_INSTANCE_NUM='${FLEXKV_INSTANCE_NUM}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.FLEXKV_SERVER_RECV_PORT='${FLEXKV_SERVER_RECV_PORT}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.FLEXKV_SERVER_LAUNCH_MODE='${FLEXKV_SERVER_LAUNCH_MODE}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.FLEXKV_ENABLE_MPS='${FLEXKV_ENABLE_MPS}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_KV_OFFLOAD_PUT_STATUS=enabled"
  )
}

setup_flexkv() {
  PYTHONPATH="${RUNTIME_PYTHONPATH}" python3 -P - <<'PY'
import flexkv
import vllm

print(f"flexkv_module={flexkv.__file__}")
print(f"vllm_module={vllm.__file__}")
PY
  echo "flexkv_cpu_cache_gb=${FLEXKV_CPU_CACHE_GB}"
  echo "flexkv_instance_num=${FLEXKV_INSTANCE_NUM}"
  echo "flexkv_server_recv_port=${FLEXKV_SERVER_RECV_PORT}"
  echo "flexkv_server_launch_mode=${FLEXKV_SERVER_LAUNCH_MODE}"
  echo "flexkv_enable_mps=${FLEXKV_ENABLE_MPS}"
  echo "vllm_abort_kv_offload=${VLLM_ABORT_KV_OFFLOAD}"
  echo "vllm_cumem_enable_shareable_handle=${VLLM_CUMEM_ENABLE_SHAREABLE_HANDLE}"
}

cleanup_flexkv() {
  return 0
}
