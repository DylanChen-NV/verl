#!/usr/bin/env bash
set -xeuo pipefail

export MODEL_PATH=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/05_claude_ws/public_models/verl/Qwen3-30B-A3B
export MCORE_MODEL_PATH=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/05_claude_ws/public_models/verl/Qwen3-30B-A3B-mcore-pp2
export VERL_COMMIT=215fa9884a3b4877d9bf2f1251b25c509a34187a
export TRAINER_TP=2 TRAINER_PP=2 TRAINER_EP=4 TRAINER_ETP=1 TRAINER_CP=1
export TRAINER_PPO_MICRO_BSZ=${TRAINER_PPO_MICRO_BSZ:-1} TRAINER_SAVE_FREQ=${TRAINER_SAVE_FREQ:--1}
export TRAIN_PROMPT_MINI_BSZ=${TRAIN_PROMPT_MINI_BSZ:-8} REQUIRE_BATCHES=${REQUIRE_BATCHES:-2}
export PLANNED_TRAINER_STEPS=${PLANNED_TRAINER_STEPS:-5} TOTAL_ROLLOUT_STEPS=${TOTAL_ROLLOUT_STEPS:-80}
export EXPECTED_LOGICAL_REQUESTS=${EXPECTED_LOGICAL_REQUESTS:-${EXPECTED_MLFLOW_TRACES:-120}}
export MINIMUM_RETRY_ATTEMPTS=${MINIMUM_RETRY_ATTEMPTS:-16}
export MAX_PROMPT_LENGTH=1024 MAX_RESPONSE_LENGTH=8192 ACTOR_PPO_MAX_TOKEN_LEN=18432 INFER_PPO_MAX_TOKEN_LEN=27648
export N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-2}
export CONCURRENT_SAMPLES_PER_REPLICA=${CONCURRENT_SAMPLES_PER_REPLICA:-32}
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-32} ROLLOUT_TP=${ROLLOUT_TP:-2}
export ROLLOUT_IGNORE_EOS=${ROLLOUT_IGNORE_EOS:-True}
export STALENESS_THRESHOLD=3.0
export RAY_CLEANUP_SETTLE_SECONDS=${RAY_CLEANUP_SETTLE_SECONDS:-5}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.70}
export STANDALONE_GPU_MEMORY_UTILIZATION=${STANDALONE_GPU_MEMORY_UTILIZATION:-0.70}
export ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-True}
export DYNAMIC_DEACTIVATE_RATIO=${DYNAMIC_DEACTIVATE_RATIO:-0.25}
export VERL_RECOMPUTE_DEACTIVATE_GRACE_S=${VERL_RECOMPUTE_DEACTIVATE_GRACE_S:-0}
export WANDB_MODE=offline
export ABC_GROUP=${ABC_GROUP:-A}
export USE_PREPARED_REPO=1
RUN_ID=${VERL_RUN_ID:-${SLURM_JOB_ID:-manual}}

ROOT=${ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/05_claude_ws}
WORKSPACE=${WORKSPACE:-${ROOT}/dynamic_flexkv_abort_dev}
REPO=${REPO:-${WORKSPACE}/src/verl}
VLLM_REPO=${VLLM_REPO:-${WORKSPACE}/src/vllm}
FLEXKV_REPO=${FLEXKV_REPO:-${WORKSPACE}/src/FlexKV}
LOGDIR=${LOGDIR:-${WORKSPACE}/logs}
PYDEPS_DIR=${PYDEPS_DIR:-${ROOT}/dynamic_recompute_h100/pydeps_cupy_npy22}
MONITORING_ROOT=${MONITORING_ROOT:-${WORKSPACE}/monitoring}
MONITORING_TOOLS_ROOT=${MONITORING_TOOLS_ROOT:-${ROOT}/dynamic_recompute_h100/monitoring}
MLFLOW_SITE=${MLFLOW_SITE:-${ROOT}/exp_verl_main_mlflow_perfetto/deps/mlflow-3.14.0}
MLFLOW_PORT=${MLFLOW_PORT:-15000}
MLFLOW_PROJECT=${MLFLOW_PROJECT:-dynamic-recompute-a80-${RUN_ID}}
MLFLOW_PATCH=${MLFLOW_PATCH:-${MONITORING_TOOLS_ROOT}/verl_dynamic_recompute_mlflow.patch}
MLFLOW_DUMP_SCRIPT=${MLFLOW_DUMP_SCRIPT:-${MONITORING_TOOLS_ROOT}/dump_dynamic_recompute_mlflow.py}


WANDB_ENTITY=${WANDB_ENTITY:-czqing422-sjtu}
WANDB_PROJECT=${WANDB_PROJECT:-verl-dynamic-recompute-measurement}
WANDB_MODE=${WANDB_MODE:-offline}
WANDB_GROUP=${WANDB_GROUP:-dynamic-recompute-dfw-h100-dapo}
WANDB_TAGS=${WANDB_TAGS:-measurement,dynamic-resource,recompute,fully-async,dfw,h100,2nodes,dapo-math-17k}

mkdir -p "${WORKSPACE}" "${LOGDIR}" "${WORKSPACE}/wandb" "${PYDEPS_DIR}" "${MONITORING_ROOT}/runs"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE WANDB_ENTITY WANDB_PROJECT WANDB_GROUP WANDB_TAGS
export WANDB_DIR=${WANDB_DIR:-${WORKSPACE}/wandb}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-${ROOT}/03_Data/verl/cache/wandb}
export HF_HOME=${HF_HOME:-${ROOT}/03_Data/verl/cache/huggingface}
export TMPDIR=/tmp
export RAY_raylet_start_wait_time_s=${RAY_raylet_start_wait_time_s:-180}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/tmp/torch_extensions_${RUN_ID}}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_USE_V1=${VLLM_USE_V1:-1}
export RAY_DEDUP_LOGS=0
export VERL_RECOMPUTE_TRACE=1
export VERL_RECOMPUTE_DEACTIVATE_GRACE_S=${VERL_RECOMPUTE_DEACTIVATE_GRACE_S:-2.0}
export MLFLOW_ENABLE_ASYNC_TRACE_LOGGING=${MLFLOW_ENABLE_ASYNC_TRACE_LOGGING:-false}
RUNTIME_PYTHONPATH="${PYDEPS_DIR}:${REPO}:${VLLM_REPO}:${FLEXKV_REPO}:${MLFLOW_SITE}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${PYDEPS_DIR}/nvidia/nccl/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
mkdir -p "${HF_HOME}" "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}"
export HOME=${WORKSPACE}/home
export CUPY_CACHE_DIR=${CUPY_CACHE_DIR:-${WORKSPACE}/cupy_cache}
mkdir -p "${HOME}" "${HOME}/.cupy" "${CUPY_CACHE_DIR}"

JOB_ID=${RUN_ID}
RANK=${SLURM_PROCID:-0}
LOCAL_RANK=${SLURM_LOCALID:-0}
MLFLOW_RUN_DIR=${MLFLOW_RUN_DIR:-${MONITORING_ROOT}/runs/${JOB_ID}}
MLFLOW_DB=${MLFLOW_DB:-${MLFLOW_RUN_DIR}/mlflow.db}
MLFLOW_ARTIFACT_ROOT=${MLFLOW_ARTIFACT_ROOT:-${MLFLOW_RUN_DIR}/artifacts}
MLFLOW_OUTPUT_DIR=${MLFLOW_OUTPUT_DIR:-${MLFLOW_RUN_DIR}/trace_dump}
export MLFLOW_TRACKING_URI="http://127.0.0.1:${MLFLOW_PORT}"
mkdir -p "${MLFLOW_RUN_DIR}" "${MLFLOW_ARTIFACT_ROOT}" "${MLFLOW_OUTPUT_DIR}"
source "${WORKSPACE}/scripts/flexkv_runtime.inc.sh"
configure_flexkv

NNODES_TRAIN=${NNODES_TRAIN:-1}
NNODES_ROLLOUT=${NNODES_ROLLOUT:-1}
NNODES_TOTAL=${NNODES_TOTAL:-$((NNODES_TRAIN + NNODES_ROLLOUT))}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
RAY_PORT=${RAY_PORT:-6379}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}
RAY_MIN_WORKER_PORT=${RAY_MIN_WORKER_PORT:-20000}
RAY_MAX_WORKER_PORT=${RAY_MAX_WORKER_PORT:-29999}
RAY_METRICS_EXPORT_PORT=${RAY_METRICS_EXPORT_PORT:-8080}
RAY_NODE_MANAGER_BASE_PORT=${RAY_NODE_MANAGER_BASE_PORT:-17000}
RAY_OBJECT_MANAGER_BASE_PORT=${RAY_OBJECT_MANAGER_BASE_PORT:-17100}
RAY_DASHBOARD_AGENT_GRPC_BASE_PORT=${RAY_DASHBOARD_AGENT_GRPC_BASE_PORT:-17200}
RAY_DASHBOARD_AGENT_LISTEN_BASE_PORT=${RAY_DASHBOARD_AGENT_LISTEN_BASE_PORT:-17300}
RAY_RUNTIME_ENV_AGENT_BASE_PORT=${RAY_RUNTIME_ENV_AGENT_BASE_PORT:-17400}
HEAD_WAIT_SECONDS=${HEAD_WAIT_SECONDS:-1800}
CLUSTER_WAIT_SECONDS=${CLUSTER_WAIT_SECONDS:-900}
export NNODES_TRAIN NNODES_ROLLOUT NNODES_TOTAL NGPUS_PER_NODE CLUSTER_WAIT_SECONDS

RAY_NODE_MANAGER_PORT=$((RAY_NODE_MANAGER_BASE_PORT + RANK))
RAY_OBJECT_MANAGER_PORT=$((RAY_OBJECT_MANAGER_BASE_PORT + RANK))
RAY_DASHBOARD_AGENT_GRPC_PORT=$((RAY_DASHBOARD_AGENT_GRPC_BASE_PORT + RANK))
RAY_DASHBOARD_AGENT_LISTEN_PORT=$((RAY_DASHBOARD_AGENT_LISTEN_BASE_PORT + RANK))
RAY_RUNTIME_ENV_AGENT_PORT=$((RAY_RUNTIME_ENV_AGENT_BASE_PORT + RANK))
HEAD_FILE="${LOGDIR}/ray_head_${JOB_ID}.txt"
DONE_FILE="${LOGDIR}/ray_done_${JOB_ID}.txt"
HEAD_LOCK_DIR="${LOGDIR}/ray_head_${JOB_ID}.lock"
REPO_READY_FILE="${LOGDIR}/repo_ready_${JOB_ID}.txt"
REPO_FAIL_FILE="${LOGDIR}/repo_failed_${JOB_ID}.txt"
RAY_TEMP_DIR="/tmp/verl_dyn_ray_${JOB_ID}"
VLLM_RPC_BASE_PATH="/tmp/verl_dyn_vllm_rpc_${JOB_ID}"
export VLLM_RPC_BASE_PATH
mkdir -p "${RAY_TEMP_DIR}" "${VLLM_RPC_BASE_PATH}"

MODEL_ID=${MODEL_ID:-Qwen/Qwen3-8B}
MODEL_PATH=${MODEL_PATH:-${ROOT}/public_models/verl/Qwen3-8B}
DATA_DIR=${DATA_DIR:-${ROOT}/public_data/verl/dapo}
TRAIN_FILE=${TRAIN_FILE:-${DATA_DIR}/dapo-math-17k.parquet}
TEST_FILE=${TEST_FILE:-${DATA_DIR}/aime-2024.parquet}
DAPO_REVISION=${DAPO_REVISION:-65877096c24ffa7abc4e4fa5edb95cf3413a5674}
EXPECTED_DAPO_SHA256=${EXPECTED_DAPO_SHA256:-534375d6bb8630d22ab46a56e11f2ffec1d288d8f7d04099bc82d68948705941}
EXPECTED_AIME_SHA256=${EXPECTED_AIME_SHA256:-12154e38a716d12db5731f9a022ae69a610c4f7d0e0dcc04e902887a686877e7}

project_name=${PROJECT_NAME:-${WANDB_PROJECT}}
base_exp_name=${EXP_NAME:-qwen3-dapo-dynamic-recompute-megatron-dfw-h100}
exp_name="${base_exp_name}-${JOB_ID}"
CKPTS_DIR=${CKPTS_DIR:-${WORKSPACE}/checkpoints/${project_name}/${exp_name}}
mkdir -p "${CKPTS_DIR}"

export WANDB_RUN_ID=${WANDB_RUN_ID:-dyn-recompute-dapo-dfw-h100-${JOB_ID}}
export WANDB_NAME=${WANDB_NAME:-${exp_name}}
export WANDB_RESUME=${WANDB_RESUME:-allow}

echo "date=$(date -Is)"
echo "hostname=$(hostname)"
echo "rank=${RANK}"
echo "local_rank=${LOCAL_RANK}"
echo "slurm_job_id=${SLURM_JOB_ID:-unset}"
echo "slurm_job_nodelist=${SLURM_JOB_NODELIST:-unset}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "workspace=${WORKSPACE}"
echo "repo=${REPO}"
echo "pydeps_dir=${PYDEPS_DIR}"
echo "model_path=${MODEL_PATH}"
echo "train_file=${TRAIN_FILE}"
echo "dapo_revision=${DAPO_REVISION}"
echo "expected_dapo_sha256=${EXPECTED_DAPO_SHA256}"
echo "test_file=${TEST_FILE}"
echo "WANDB_MODE=${WANDB_MODE}"
echo "WANDB_RUN_ID=${WANDB_RUN_ID}"
echo "RAY_TEMP_DIR=${RAY_TEMP_DIR}"
echo "VLLM_RPC_BASE_PATH=${VLLM_RPC_BASE_PATH}"
echo "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
echo "TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR}"
echo "RUNTIME_PYTHONPATH=${RUNTIME_PYTHONPATH}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

nvidia-smi -L
if [[ "${USE_PREPARED_REPO}" != "1" ]]; then

if [[ "${RANK}" == "0" ]]; then
  rm -f "${REPO_READY_FILE}" "${REPO_FAIL_FILE}"
  if [[ ! -d "${REPO}/.git" ]]; then
    git clone --depth 1 --branch dynamic_resource https://github.com/meituan-search/verl.git "${REPO}" || {
      status=$?
      echo "git clone failed with status ${status}" > "${REPO_FAIL_FILE}"
      exit "${status}"
    }
  fi
  git -C "${REPO}" fetch --depth 1 origin "${VERL_COMMIT}" || {
    status=$?
    echo "git fetch ${VERL_COMMIT} failed with status ${status}" > "${REPO_FAIL_FILE}"
    exit "${status}"
  }
  git -C "${REPO}" checkout dynamic_resource || {
    status=$?
    echo "git checkout failed with status ${status}" > "${REPO_FAIL_FILE}"
    exit "${status}"
  }
  git -C "${REPO}" reset --hard "${VERL_COMMIT}" || {
    status=$?
    echo "git reset ${VERL_COMMIT} failed with status ${status}" > "${REPO_FAIL_FILE}"
    exit "${status}"
  }
  git -C "${REPO}" rev-parse HEAD > "${REPO_READY_FILE}" || {
    status=$?
    echo "git rev-parse failed with status ${status}" > "${REPO_FAIL_FILE}"
    exit "${status}"
  }
else
  for _ in $(seq 1 "${HEAD_WAIT_SECONDS}"); do
    [[ -s "${REPO_FAIL_FILE}" ]] && break
    [[ -s "${REPO_READY_FILE}" ]] && break
    sleep 1
  done
  if [[ -s "${REPO_FAIL_FILE}" ]]; then
    echo "repo setup failed on rank0"
    cat "${REPO_FAIL_FILE}"
    exit 1
  fi
  test -s "${REPO_READY_FILE}"
fi

cd "${REPO}"
git rev-parse --short HEAD

PATCH_READY_FILE="${LOGDIR}/source_patch_ready_${JOB_ID}.txt"
PATCH_FAIL_FILE="${LOGDIR}/source_patch_failed_${JOB_ID}.txt"
if [[ "${RANK}" == "0" ]]; then
  rm -f "${PATCH_READY_FILE}" "${PATCH_FAIL_FILE}"
  (
python3 -S - <<'PY_PATCH'
from pathlib import Path

path = Path("verl/models/transformers/monkey_patch.py")
text = path.read_text()
old = """    if is_trl_available():
        from trl import AutoModelForCausalLMWithValueHead  # type: ignore

        def state_dict(self, *args, **kwargs):
            return torch.nn.Module.state_dict(self, *args, **kwargs)

        AutoModelForCausalLMWithValueHead.state_dict = state_dict
        print("Monkey patch state_dict in AutoModelForCausalLMWithValueHead. ")
"""
new = """    if is_trl_available():
        try:
            from trl import AutoModelForCausalLMWithValueHead  # type: ignore
        except ImportError:
            try:
                from trl.models.modeling_value_head import AutoModelForCausalLMWithValueHead  # type: ignore
            except ImportError:
                AutoModelForCausalLMWithValueHead = None

        if AutoModelForCausalLMWithValueHead is not None:
            def state_dict(self, *args, **kwargs):
                return torch.nn.Module.state_dict(self, *args, **kwargs)

            AutoModelForCausalLMWithValueHead.state_dict = state_dict
            print("Monkey patch state_dict in AutoModelForCausalLMWithValueHead. ")
        else:
            print("Skip monkey patch state_dict in AutoModelForCausalLMWithValueHead: not available in installed trl.")
"""
if old in text:
    path.write_text(text.replace(old, new))
    print("Applied TRL value-head import compatibility patch")
elif "from trl.models.modeling_value_head import AutoModelForCausalLMWithValueHead" in text:
    print("TRL value-head import compatibility patch already applied")
else:
    raise SystemExit("TRL value-head monkey patch target not found")
PY_PATCH

python3 -S - <<'PY_PATCH_VLLM'
from pathlib import Path

sitecustomize = Path("sitecustomize.py")
marker = "# verl dynamic-resource smoke compatibility shims"
snippet = '''# verl dynamic-resource smoke compatibility shims
try:
    import transformers.configuration_utils as _verl_configuration_utils
except Exception:
    _verl_configuration_utils = None

if _verl_configuration_utils is not None and not hasattr(_verl_configuration_utils, "ALLOWED_LAYER_TYPES"):
    _verl_configuration_utils.ALLOWED_LAYER_TYPES = [
        "full_attention",
        "sliding_attention",
        "chunked_attention",
    ]
'''
site_text = sitecustomize.read_text() if sitecustomize.exists() else ""
if marker not in site_text:
    sitecustomize.write_text(snippet + ("\n" + site_text if site_text else ""))
    print("Applied sitecustomize transformers ALLOWED_LAYER_TYPES compatibility patch")
else:
    print("sitecustomize transformers ALLOWED_LAYER_TYPES compatibility patch already applied")

path = Path("verl/third_party/vllm/__init__.py")
text = path.read_text()
old = """VLLM_SLEEP_LEVEL = 1

if package_version is None:
"""
new = """VLLM_SLEEP_LEVEL = 1


def _ensure_transformers_allowed_layer_types():
    try:
        import transformers.configuration_utils as configuration_utils
    except Exception:
        return

    if not hasattr(configuration_utils, "ALLOWED_LAYER_TYPES"):
        configuration_utils.ALLOWED_LAYER_TYPES = [
            "full_attention",
            "sliding_attention",
            "chunked_attention",
        ]


_ensure_transformers_allowed_layer_types()

if package_version is None:
"""
if old in text:
    path.write_text(text.replace(old, new))
    print("Applied transformers ALLOWED_LAYER_TYPES compatibility patch")
elif "_ensure_transformers_allowed_layer_types" in text:
    print("transformers ALLOWED_LAYER_TYPES compatibility patch already applied")
else:
    raise SystemExit("vLLM transformers compatibility patch target not found")
PY_PATCH_VLLM

python3 -S - <<'PY_PATCH_NCCL'
from pathlib import Path

path = Path("verl/checkpoint_engine/nccl_checkpoint_engine.py")
text = path.read_text()
changed = False

old = """import asyncio
import logging
"""
new = """from __future__ import annotations

import asyncio
import logging
"""
if old in text and "from __future__ import annotations" not in text:
    text = text.replace(old, new, 1)
    changed = True

old = """with patch("importlib.metadata.distributions", return_value=[]):
    import cupy as cp
"""
new = """with patch("importlib.metadata.distributions", return_value=[]):
    try:
        import cupy as cp
    except ImportError:
        cp = None
"""
if old in text:
    text = text.replace(old, new, 1)
    changed = True

old = """        if self.is_master:
            self.send_buf = cp.zeros(self.bucket_size, dtype=cp.uint8)
            self.recv_buf = cp.zeros(self.bucket_size, dtype=cp.uint8)
        else:
            self.send_buf = torch.zeros(self.bucket_size, dtype=torch.uint8, device="cuda")
            self.recv_buf = torch.zeros(self.bucket_size, dtype=torch.uint8, device="cuda")
"""
new = """        if self.is_master and cp is not None:
            self.send_buf = cp.zeros(self.bucket_size, dtype=cp.uint8)
            self.recv_buf = cp.zeros(self.bucket_size, dtype=cp.uint8)
        else:
            self.send_buf = torch.zeros(self.bucket_size, dtype=torch.uint8, device="cuda")
            self.recv_buf = torch.zeros(self.bucket_size, dtype=torch.uint8, device="cuda")
"""
if old in text:
    text = text.replace(old, new, 1)
    changed = True

old = """            send_buf[offset : offset + tensor_meta.chunk_size] = cp.asarray(chunk)
"""
new = """            if cp is not None and not isinstance(send_buf, torch.Tensor):
                send_buf[offset : offset + tensor_meta.chunk_size] = cp.asarray(chunk)
            else:
                send_buf[offset : offset + tensor_meta.chunk_size] = chunk
"""
if old in text:
    text = text.replace(old, new, 1)
    changed = True

if changed:
    path.write_text(text)
    print("Applied NCCL checkpoint engine cupy fallback patch")
elif "cp = None" in text and "not isinstance(send_buf, torch.Tensor)" in text:
    print("NCCL checkpoint engine cupy fallback patch already applied")
else:
    raise SystemExit("NCCL checkpoint engine cupy fallback patch target not found")
PY_PATCH_NCCL

  ) || {
    status=$?
    echo "source patch failed with status ${status}" > "${PATCH_FAIL_FILE}"
    exit "${status}"
  }
  echo ready > "${PATCH_READY_FILE}"
else
  for _ in $(seq 1 "${HEAD_WAIT_SECONDS}"); do
    [[ -s "${PATCH_FAIL_FILE}" ]] && break
    [[ -s "${PATCH_READY_FILE}" ]] && break
    sleep 1
  done
  if [[ -s "${PATCH_FAIL_FILE}" ]]; then
    echo "source patch failed on rank0"
    cat "${PATCH_FAIL_FILE}"
    exit 1
  fi
  test -s "${PATCH_READY_FILE}"
fi

RECOMPUTE_PATCH_READY_FILE="${LOGDIR}/recompute_patch_ready_${JOB_ID}.txt"
RECOMPUTE_PATCH_FAIL_FILE="${LOGDIR}/recompute_patch_failed_${JOB_ID}.txt"
if [[ "${RANK}" == "0" ]]; then
  rm -f "${RECOMPUTE_PATCH_READY_FILE}" "${RECOMPUTE_PATCH_FAIL_FILE}"
  (
python3 -S - <<'PY_PATCH_RECOMPUTE_VERL'
from pathlib import Path


client_path = Path("verl/workers/rollout/llm_server.py")
text = client_path.read_text()
changed = False

if "def _verl_recompute_event(" not in text:
    imports = "import asyncio\nimport logging\nimport os\n"
    new_imports = "import asyncio\nimport json\nimport logging\nimport os\nimport time\n"
    if imports not in text:
        raise SystemExit("llm_server import target not found")
    text = text.replace(imports, new_imports, 1)

    target = 'logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))\n\n'
    helper = '''logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _verl_recompute_event(phase: str, **fields: Any) -> None:
    if os.getenv("VERL_RECOMPUTE_TRACE", "0") != "1":
        return
    payload = {
        "phase": phase,
        "monotonic_ns": time.monotonic_ns(),
        "wall_ns": time.time_ns(),
        **fields,
    }
    print("VERL_RECOMPUTE_EVENT " + json.dumps(payload, sort_keys=True), flush=True)

'''
    if target not in text:
        raise SystemExit("llm_server event helper target not found")
    text = text.replace(target, helper, 1)
    changed = True

old = '                request_id=uuid4().hex,  # use new request_id for each turn\n'
new = '''                request_id=(
                    request_id
                    if "__verl_recompute_attempt_" in request_id
                    else uuid4().hex
                ),
'''
if old in text:
    text = text.replace(old, new, 1)
    changed = True
elif '"__verl_recompute_attempt_" in request_id' not in text:
    raise SystemExit("llm_server engine request-id preservation target not found")

old = '''        server_id, server = await self._acquire_server(request_id)
        try:
'''
new = '''        _verl_recompute_event(
            "CLIENT_ENTER",
            logical_request_id=str(request_id),
            prompt_tokens=len(prompt_ids),
        )
        lb_status = await self._load_balancer.get_status.remote()
        _verl_recompute_event(
            "LB_BEFORE_ACQUIRE",
            logical_request_id=str(request_id),
            lb_status=lb_status,
        )
        server_id, server = await self._acquire_server(request_id)
        _verl_recompute_event(
            "LB_AFTER_ACQUIRE",
            logical_request_id=str(request_id),
            server_id=str(server_id),
        )
        try:
'''
if old in text:
    text = text.replace(old, new, 1)
    changed = True
elif '"LB_BEFORE_ACQUIRE"' not in text:
    raise SystemExit("llm_server acquire instrumentation target not found")

old = '''            output: TokenOutput = await server.generate.remote(
'''
new = '''            _verl_recompute_event(
                "SERVER_RPC_BEGIN",
                logical_request_id=str(request_id),
                server_id=str(server_id),
            )
            output: TokenOutput = await server.generate.remote(
'''
if old in text:
    text = text.replace(old, new, 1)
    changed = True
elif '"SERVER_RPC_BEGIN"' not in text:
    raise SystemExit("llm_server server rpc begin target not found")

old = '''            return output
        finally:
            self._release_server(server_id)
'''
new = '''            _verl_recompute_event(
                "SERVER_RPC_END",
                logical_request_id=str(request_id),
                server_id=str(server_id),
                output_tokens=len(output.token_ids),
            )
            return output
        finally:
            self._release_server(server_id)
'''
if old in text:
    text = text.replace(old, new, 1)
    changed = True
elif '"SERVER_RPC_END"' not in text:
    raise SystemExit("llm_server server rpc end target not found")

old = '''        min_global_steps, max_global_steps = None, None

        while True:
            # 1. generate tokens
            output = await super().generate(
                request_id=request_id,
                prompt_ids=prompt_ids + final_output.token_ids,
'''
new = '''        min_global_steps, max_global_steps = None, None
        attempt_id = 0

        while True:
            engine_request_id = (
                request_id
                if attempt_id == 0
                else f"{request_id}__verl_recompute_attempt_{attempt_id}"
            )
            if attempt_id > 0:
                _verl_recompute_event(
                    "RETRY_SUBMIT",
                    logical_request_id=str(request_id),
                    engine_request_id=str(engine_request_id),
                    attempt_id=attempt_id,
                    prompt_tokens=len(prompt_ids),
                    partial_tokens=len(final_output.token_ids),
                    retry_prefix_tokens=len(prompt_ids) + len(final_output.token_ids),
                )

            # 1. generate tokens
            output = await super().generate(
                request_id=engine_request_id,
                prompt_ids=prompt_ids + final_output.token_ids,
'''
if old in text:
    text = text.replace(old, new, 1)
    changed = True
elif "engine_request_id = (" not in text:
    raise SystemExit("llm_server retry request-id target not found")

old = '''            if output.stop_reason not in ("aborted", "abort") or not should_retry:
                break

            await asyncio.sleep(1)
'''
new = '''            if output.stop_reason not in ("aborted", "abort") or not should_retry:
                break

            if attempt_id > 0:
                _verl_recompute_event(
                    "CLIENT_REABORT",
                    logical_request_id=str(request_id),
                    engine_request_id=str(engine_request_id),
                    attempt_id=attempt_id,
                    retry_prefix_tokens=len(prompt_ids) + len(final_output.token_ids),
                )
            attempt_id += 1
            await asyncio.sleep(1)
'''
if old in text:
    text = text.replace(old, new, 1)
    changed = True
elif "CLIENT_REABORT" not in text:
    raise SystemExit("llm_server retry increment target not found")

if changed:
    client_path.write_text(text)
    print("Applied VERL llm_server recompute instrumentation")
else:
    print("VERL llm_server recompute instrumentation already applied")


agent_path = Path("verl/experimental/agent_loop/agent_loop.py")
agent_text = agent_path.read_text()
old = '''                selected_reward_loop_worker_handle = random.choice(self.reward_loop_worker_handles)
                result = await selected_reward_loop_worker_handle.compute_score.remote(data)
'''
new = '''                selected_reward_loop_worker_handle = random.choice(self.reward_loop_worker_handles)
                print(
                    f"VERL_REWARD_EVENT phase=AGENT_REWARD_BEGIN pid={__import__('os').getpid()} "
                    f"handle={selected_reward_loop_worker_handle}",
                    flush=True,
                )
                result = await selected_reward_loop_worker_handle.compute_score.remote(data)
                print(
                    f"VERL_REWARD_EVENT phase=AGENT_REWARD_END pid={__import__('os').getpid()}",
                    flush=True,
                )
'''
if old in agent_text:
    agent_text = agent_text.replace(old, new, 1)
    agent_path.write_text(agent_text)
elif "phase=AGENT_REWARD_BEGIN" not in agent_text:
    raise SystemExit("agent reward instrumentation target not found")

reward_path = Path("verl/experimental/reward_loop/reward_manager/naive.py")
reward_text = reward_path.read_text()
old = '''    async def run_single(self, data: DataProto) -> dict:
        data = data[-1:]  # for multi-sequence outputs, we only compute reward based on the last sequence
'''
new = '''    async def run_single(self, data: DataProto) -> dict:
        print(
            f"VERL_REWARD_EVENT phase=REWARD_RUN_BEGIN pid={__import__('os').getpid()}",
            flush=True,
        )
        data = data[-1:]  # for multi-sequence outputs, we only compute reward based on the last sequence
'''
if old in reward_text:
    reward_text = reward_text.replace(old, new, 1)
elif "phase=REWARD_RUN_BEGIN" not in reward_text:
    raise SystemExit("reward run begin target not found")

old = '''        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )

        extra_reward_kwargs = (
'''
new = '''        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )
        print(
            f"VERL_REWARD_EVENT phase=REWARD_DECODE_END pid={__import__('os').getpid()}",
            flush=True,
        )

        extra_reward_kwargs = (
'''
if old in reward_text:
    reward_text = reward_text.replace(old, new, 1)
elif "phase=REWARD_DECODE_END" not in reward_text:
    raise SystemExit("reward decode instrumentation target not found")

old = '''        reward = score

        return {"reward_score": reward, "reward_extra_info": reward_extra_info}
'''
new = '''        reward = score
        print(
            f"VERL_REWARD_EVENT phase=REWARD_SCORE_END pid={__import__('os').getpid()} score={reward}",
            flush=True,
        )

        return {"reward_score": reward, "reward_extra_info": reward_extra_info}
'''
if old in reward_text:
    reward_text = reward_text.replace(old, new, 1)
elif "phase=REWARD_SCORE_END" not in reward_text:
    raise SystemExit("reward score instrumentation target not found")
reward_path.write_text(reward_text)

# Final boundaries: per-trajectory postprocess, worker gather/batch, manager RPC, and queue put.
agent_text = agent_path.read_text()
old = '''            output: AgentLoopOutput = await agent_loop.run(sampling_params, **kwargs)
            return await self._agent_loop_postprocess(output, trajectory["validate"], **kwargs)
'''
new = '''            output: AgentLoopOutput = await agent_loop.run(sampling_params, **kwargs)
            print(f"VERL_PIPELINE_EVENT phase=TRAJECTORY_POSTPROCESS_BEGIN pid={__import__('os').getpid()}", flush=True)
            result = await self._agent_loop_postprocess(output, trajectory["validate"], **kwargs)
            print(f"VERL_PIPELINE_EVENT phase=TRAJECTORY_POSTPROCESS_END pid={__import__('os').getpid()}", flush=True)
            return result
'''
if old in agent_text:
    agent_text = agent_text.replace(old, new, 1)
elif "phase=TRAJECTORY_POSTPROCESS_BEGIN" not in agent_text:
    raise SystemExit("trajectory postprocess instrumentation target not found")

old = '''        outputs = await asyncio.gather(*tasks)

        output = self._postprocess(
            outputs, input_non_tensor_batch=batch.non_tensor_batch, validate=batch.meta_info.get("validate", False)
        )
        return output
'''
new = '''        outputs = await asyncio.gather(*tasks)
        print(f"VERL_PIPELINE_EVENT phase=WORKER_GATHER_END pid={__import__('os').getpid()} outputs={len(outputs)}", flush=True)

        output = self._postprocess(
            outputs, input_non_tensor_batch=batch.non_tensor_batch, validate=batch.meta_info.get("validate", False)
        )
        print(f"VERL_PIPELINE_EVENT phase=WORKER_BATCH_END pid={__import__('os').getpid()} batch={len(output)}", flush=True)
        return output
'''
if old in agent_text:
    agent_text = agent_text.replace(old, new, 1)
elif "phase=WORKER_GATHER_END" not in agent_text:
    raise SystemExit("worker gather instrumentation target not found")
agent_path.write_text(agent_text)

rollouter_path = Path("verl/experimental/fully_async_policy/fully_async_rollouter.py")
rollouter_text = rollouter_path.read_text()
old = '''        output_future = worker.generate_sequences.remote(prompts)
        return await asyncio.wrap_future(output_future.future())
'''
new = '''        print(f"VERL_PIPELINE_EVENT phase=MANAGER_RPC_BEGIN pid={__import__('os').getpid()}", flush=True)
        output_future = worker.generate_sequences.remote(prompts)
        output = await asyncio.wrap_future(output_future.future())
        print(f"VERL_PIPELINE_EVENT phase=MANAGER_RPC_END pid={__import__('os').getpid()} batch={len(output)}", flush=True)
        return output
'''
if old in rollouter_text:
    rollouter_text = rollouter_text.replace(old, new, 1)
elif "phase=MANAGER_RPC_BEGIN" not in rollouter_text:
    raise SystemExit("manager rpc instrumentation target not found")

old = '''        ret = await self.async_rollout_manager.generate_sequences_single(rollout_sample.full_batch)

        rollout_sample.full_batch = ret
'''
new = '''        print(f"VERL_PIPELINE_EVENT phase=SAMPLE_GENERATE_BEGIN pid={__import__('os').getpid()} sample={rollout_sample.sample_id}", flush=True)
        ret = await self.async_rollout_manager.generate_sequences_single(rollout_sample.full_batch)
        print(f"VERL_PIPELINE_EVENT phase=SAMPLE_GENERATE_END pid={__import__('os').getpid()} sample={rollout_sample.sample_id}", flush=True)

        rollout_sample.full_batch = ret
'''
if old in rollouter_text:
    rollouter_text = rollouter_text.replace(old, new, 1)
elif "phase=SAMPLE_GENERATE_BEGIN" not in rollouter_text:
    raise SystemExit("sample generate instrumentation target not found")

old = '''        success = await self.message_queue_client.put_sample(
            sample=ray.cloudpickle.dumps(rollout_sample),
        )
'''
new = '''        print(f"VERL_PIPELINE_EVENT phase=QUEUE_PUT_BEGIN pid={__import__('os').getpid()} sample={rollout_sample.sample_id}", flush=True)
        success = await self.message_queue_client.put_sample(
            sample=ray.cloudpickle.dumps(rollout_sample),
        )
        print(f"VERL_PIPELINE_EVENT phase=QUEUE_PUT_END pid={__import__('os').getpid()} sample={rollout_sample.sample_id} success={success}", flush=True)
'''
if old in rollouter_text:
    rollouter_text = rollouter_text.replace(old, new, 1)
elif "phase=QUEUE_PUT_BEGIN" not in rollouter_text:
    raise SystemExit("queue put instrumentation target not found")
rollouter_path.write_text(rollouter_text)

controller_path = Path(
    "verl/experimental/fully_async_policy/dynamic_scaling/dynamic_resource_controller.py"
)
text = controller_path.read_text()
changed = False
if "import json\n" not in text:
    target = "import time\n"
    if target not in text:
        raise SystemExit("controller json import target not found")
    text = text.replace(target, "import json\nimport time\n", 1)
    changed = True

old = '''        print(f"[DynamicResourceController] Deactivating hybrid replicas at step {global_steps}")
        start = time.time()
'''
new = '''        print(f"[DynamicResourceController] Deactivating hybrid replicas at step {global_steps}")
        start = time.time()
        grace_s = float(__import__("os").getenv("VERL_RECOMPUTE_DEACTIVATE_GRACE_S", "0"))
        if grace_s > 0:
            print(
                f"[DynamicResourceController] Waiting {grace_s:.2f}s before deactivation "
                "to observe in-flight hybrid requests",
                flush=True,
            )
            await __import__("asyncio").sleep(grace_s)
        dynamic_cycle_id = self.deactivate_count + 1
        if __import__("os").getenv("VERL_RECOMPUTE_TRACE", "0") == "1":
            print(
                "VERL_RECOMPUTE_EVENT "
                + json.dumps(
                    {
                        "phase": "CYCLE_START",
                        "dynamic_cycle_id": dynamic_cycle_id,
                        "global_steps": global_steps,
                        "monotonic_ns": time.monotonic_ns(),
                        "wall_ns": time.time_ns(),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
'''
if old in text:
    text = text.replace(old, new, 1)
    changed = True
elif '"phase": "CYCLE_START"' not in text:
    raise SystemExit("controller cycle start target not found")

old = '''        self._hybrid_active = False
        self.deactivate_count += 1
        print(
'''
new = '''        self._hybrid_active = False
        self.deactivate_count += 1
        if __import__("os").getenv("VERL_RECOMPUTE_TRACE", "0") == "1":
            print(
                "VERL_RECOMPUTE_EVENT "
                + json.dumps(
                    {
                        "phase": "CYCLE_END",
                        "dynamic_cycle_id": dynamic_cycle_id,
                        "global_steps": global_steps,
                        "monotonic_ns": time.monotonic_ns(),
                        "wall_ns": time.time_ns(),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        print(
'''
if old in text:
    text = text.replace(old, new, 1)
    changed = True
elif '"phase": "CYCLE_END"' not in text:
    raise SystemExit("controller cycle end target not found")

if changed:
    controller_path.write_text(text)
    print("Applied dynamic cycle instrumentation")
else:
    print("Dynamic cycle instrumentation already applied")


server_path = Path("verl/workers/rollout/vllm_rollout/vllm_async_server.py")
text = server_path.read_text()
old = '''        os.environ["VERL_REPLICA_RANK"] = str(replica_rank)
        # Forward the Ray job id into the vLLM worker subprocess so the
'''
new = '''        os.environ["VERL_REPLICA_RANK"] = str(replica_rank)
        os.environ["VERL_ROLLOUT_NODE_RANK"] = str(node_rank)
        os.environ["VERL_ROLLOUT_MODE"] = str(getattr(rollout_mode, "value", rollout_mode))
        # Forward the Ray job id into the vLLM worker subprocess so the
'''
if old in text:
    server_path.write_text(text.replace(old, new, 1))
    print("Applied rollout replica provenance environment patch")
elif "VERL_ROLLOUT_NODE_RANK" in text:
    print("Rollout replica provenance environment patch already applied")
else:
    raise SystemExit("vLLM server provenance target not found")

# vLLM exposes first-scheduled and first-token timestamps on RequestOutput.
# Instrument the retry path at the server boundary so the measurement remains
# stable even when scheduler internals change between development builds.
text = server_path.read_text()
changed = False
if "def _verl_recompute_server_event(" not in text:
    if "import socket\n" not in text:
        text = text.replace("import os\n", "import os\nimport socket\nimport time\n", 1)
    target = "logger.setLevel(logging.INFO)\n\n"
    helper = '''logger.setLevel(logging.INFO)


def _verl_recompute_server_event(phase: str, request_id: str, monotonic_ns: int, **fields) -> None:
    marker = "__verl_recompute_attempt_"
    if os.getenv("VERL_RECOMPUTE_TRACE", "0") != "1" or marker not in request_id:
        return
    logical_request_id, raw_attempt = request_id.rsplit(marker, 1)
    try:
        attempt_id = int(raw_attempt)
    except ValueError:
        return
    payload = {
        "phase": phase,
        "logical_request_id": logical_request_id,
        "engine_request_id": request_id,
        "attempt_id": attempt_id,
        "monotonic_ns": monotonic_ns,
        "wall_ns": time.time_ns(),
        "node_id": socket.gethostname(),
        "replica_id": os.getenv("VERL_REPLICA_RANK", "unknown"),
        "rollout_node_rank": os.getenv("VERL_ROLLOUT_NODE_RANK", "unknown"),
        "rollout_mode": os.getenv("VERL_ROLLOUT_MODE", "unknown"),
        **fields,
    }
    print("VERL_RECOMPUTE_EVENT " + json.dumps(payload, sort_keys=True), flush=True)

'''
    if target not in text:
        raise SystemExit("vLLM server event helper target not found")
    text = text.replace(target, helper, 1)
    changed = True

old = '''        generator = self.engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
            lora_request=lora_request,
            priority=priority,
        )

        # Get final response
        final_res: Optional[RequestOutput] = None
        async for output in generator:
            final_res = output
'''
new = '''        retry_request = "__verl_recompute_attempt_" in request_id
        submit_ns = time.monotonic_ns()
        if retry_request:
            _verl_recompute_server_event(
                "SUBMIT",
                request_id,
                submit_ns,
                retry_prefix_tokens=len(prompt_ids),
                measurement_source="enginecore_scheduler",
            )
        generator = self.engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
            lora_request=lora_request,
            priority=priority,
        )

        # First output closes the queue-excluded recovery interval whose start
        # is emitted by EngineCore when the request is first scheduled.
        final_res: Optional[RequestOutput] = None
        first_output_seen = False
        async for output in generator:
            final_res = output
            if retry_request and not first_output_seen:
                first_output_seen = True
                observed_ns = time.monotonic_ns()
                event_fields = {
                    "retry_prefix_tokens": len(prompt_ids),
                    "measurement_source": "enginecore_scheduler",
                }
                phase = "FIRST_OUTPUT" if output.outputs else "REABORT"
                _verl_recompute_server_event(phase, request_id, observed_ns, **event_fields)
'''
if old in text:
    text = text.replace(old, new, 1)
    changed = True
elif "measurement_source" not in text:
    raise SystemExit("vLLM server generation timing target not found")

if changed:
    server_path.write_text(text)
    print("Applied vLLM server submit/first-output instrumentation")
else:
    print("vLLM server submit/first-output instrumentation already applied")
PY_PATCH_RECOMPUTE_VERL
  ) || {
    status=$?
    echo "recompute VERL patch failed with status ${status}" > "${RECOMPUTE_PATCH_FAIL_FILE}"
    exit "${status}"
  }
  echo ready > "${RECOMPUTE_PATCH_READY_FILE}"
else
  for _ in $(seq 1 "${HEAD_WAIT_SECONDS}"); do
    [[ -s "${RECOMPUTE_PATCH_FAIL_FILE}" ]] && break
    [[ -s "${RECOMPUTE_PATCH_READY_FILE}" ]] && break
    sleep 1
  done
  if [[ -s "${RECOMPUTE_PATCH_FAIL_FILE}" ]]; then
    cat "${RECOMPUTE_PATCH_FAIL_FILE}"
    exit 1
  fi
  test -s "${RECOMPUTE_PATCH_READY_FILE}"
fi

MLFLOW_PATCH_READY_FILE="${LOGDIR}/mlflow_patch_ready_${JOB_ID}.txt"
MLFLOW_PATCH_FAIL_FILE="${LOGDIR}/mlflow_patch_failed_${JOB_ID}.txt"
if [[ "${RANK}" == "0" ]]; then
  rm -f "${MLFLOW_PATCH_READY_FILE}" "${MLFLOW_PATCH_FAIL_FILE}"
  test -s "${MLFLOW_PATCH}"
  git apply --check "${MLFLOW_PATCH}" && git apply "${MLFLOW_PATCH}" || {
    status=$?
    echo "MLflow patch failed with status ${status}" > "${MLFLOW_PATCH_FAIL_FILE}"
    exit "${status}"
  }
  sha256sum "${MLFLOW_PATCH}" | awk '{print $1}' > "${MLFLOW_PATCH_READY_FILE}"
else
  for _ in $(seq 1 "${HEAD_WAIT_SECONDS}"); do
    [[ -s "${MLFLOW_PATCH_FAIL_FILE}" ]] && break
    [[ -s "${MLFLOW_PATCH_READY_FILE}" ]] && break
    sleep 1
  done
  if [[ -s "${MLFLOW_PATCH_FAIL_FILE}" ]]; then
    cat "${MLFLOW_PATCH_FAIL_FILE}"
    exit 1
  fi
  test -s "${MLFLOW_PATCH_READY_FILE}"
fi
export VERL_PATCH_SHA256
VERL_PATCH_SHA256=$(cat "${MLFLOW_PATCH_READY_FILE}")
else
  cd "${REPO}"
  if [[ -d "${REPO}/.git" ]]; then
    export VERL_PATCH_SHA256="prepared-$(git diff | sha256sum | awk '{print $1}')"
  else
    export VERL_PATCH_SHA256="prepared-synced-tree"
  fi
fi

echo "skip legacy vLLM recompute instrumentation for FlexKV smoke"
DATA_READY_FILE="${LOGDIR}/data_ready_${JOB_ID}.txt"
DATA_FAIL_FILE="${LOGDIR}/data_failed_${JOB_ID}.txt"
if [[ "${RANK}" == "0" ]]; then
  rm -f "${DATA_READY_FILE}" "${DATA_FAIL_FILE}"
  (
    mkdir -p "$(dirname "${MODEL_PATH}")"
    if ! PYTHONPATH="${PYDEPS_DIR}" python3 -P - <<PY_DATA_DEPS; then
import pyarrow  # noqa: F401
import huggingface_hub  # noqa: F401
PY_DATA_DEPS
      python3 -P -m pip install --no-cache-dir --target "${PYDEPS_DIR}" pyarrow huggingface_hub
    fi
    PYTHONPATH="${PYDEPS_DIR}" python3 -P - <<PY_MODEL_DATA
from pathlib import Path
import hashlib
import pyarrow.parquet as pq

model_path = Path("${MODEL_PATH}")
config_path = model_path / "config.json"
if not config_path.is_file():
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="${MODEL_ID}",
        local_dir=str(model_path),
        local_dir_use_symlinks=False,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.onnx"],
    )

if not config_path.is_file():
    raise SystemExit(f"missing model config at {config_path}")

expected = {
    Path("${TRAIN_FILE}"): "${EXPECTED_DAPO_SHA256}",
    Path("${TEST_FILE}"): "${EXPECTED_AIME_SHA256}",
}
required_columns = {"data_source", "prompt", "ability", "reward_model", "extra_info"}
for path, expected_sha in expected.items():
    if not path.is_file():
        raise SystemExit(f"missing dataset file: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_sha and digest != expected_sha:
        raise SystemExit(f"dataset checksum mismatch: {path} actual={digest} expected={expected_sha}")
    parquet = pq.ParquetFile(path)
    columns = set(parquet.schema_arrow.names)
    missing = required_columns - columns
    if missing:
        raise SystemExit(f"dataset schema missing {sorted(missing)} in {path}")
    print(
        f"dataset={path} rows={parquet.metadata.num_rows} "
        f"row_groups={parquet.metadata.num_row_groups} sha256={digest}"
    )

print(f"dapo_revision=${DAPO_REVISION}")
print(f"model_config={config_path.is_file()}")
print(f"train_file=${TRAIN_FILE}")
print(f"test_file=${TEST_FILE}")
PY_MODEL_DATA
  ) || {
    status=$?
    echo "model/data setup failed with status ${status}" > "${DATA_FAIL_FILE}"
    exit "${status}"
  }
  echo ready > "${DATA_READY_FILE}"
else
  for _ in $(seq 1 "${HEAD_WAIT_SECONDS}"); do
    [[ -s "${DATA_FAIL_FILE}" ]] && break
    [[ -s "${DATA_READY_FILE}" ]] && break
    sleep 1
  done
  if [[ -s "${DATA_FAIL_FILE}" ]]; then
    echo "model/data setup failed on rank0"
    cat "${DATA_FAIL_FILE}"
    exit 1
  fi
  test -s "${DATA_READY_FILE}"
fi

test -s "${MODEL_PATH}/config.json"
find "${MODEL_PATH}" -maxdepth 1 -name "*.safetensors" -print -quit | grep -q .
test -s "${TRAIN_FILE}"
test -s "${TEST_FILE}"

echo "skip editable install; repo is loaded through runtime PYTHONPATH"
TENSOR_READY_FILE="${LOGDIR}/tensordict_ready_${JOB_ID}.txt"
TENSOR_FAIL_FILE="${LOGDIR}/tensordict_failed_${JOB_ID}.txt"
if [[ "${RANK}" == "0" ]]; then
  rm -f "${TENSOR_READY_FILE}" "${TENSOR_FAIL_FILE}"
  (
    if ! PYTHONPATH="${PYDEPS_DIR}" python3 -P - <<'PY_TENSOR_CHECK'; then
import tensordict  # noqa: F401
PY_TENSOR_CHECK
      python3 -P -m pip install --no-cache-dir --no-deps --target "${PYDEPS_DIR}" "tensordict==0.10.0"
    fi
  ) || {
    status=$?
    echo "tensordict setup failed with status ${status}" > "${TENSOR_FAIL_FILE}"
    exit "${status}"
  }
  echo ready > "${TENSOR_READY_FILE}"
else
  for _ in $(seq 1 "${HEAD_WAIT_SECONDS}"); do
    [[ -s "${TENSOR_FAIL_FILE}" ]] && break
    [[ -s "${TENSOR_READY_FILE}" ]] && break
    sleep 1
  done
  if [[ -s "${TENSOR_FAIL_FILE}" ]]; then
    echo "tensordict setup failed on rank0"
    cat "${TENSOR_FAIL_FILE}"
    exit 1
  fi
  test -s "${TENSOR_READY_FILE}"
fi

CUPY_READY_FILE="${LOGDIR}/cupy_ready_${JOB_ID}.txt"
CUPY_FAIL_FILE="${LOGDIR}/cupy_failed_${JOB_ID}.txt"
if [[ "${RANK}" == "0" ]]; then
  rm -f "${CUPY_READY_FILE}" "${CUPY_FAIL_FILE}"
  (
    if ! PYTHONPATH="${PYDEPS_DIR}" python3 -P - <<'PY_CUPY_IMPORT'; then
import numpy as np
import cupy  # noqa: F401
from cupy.cuda.nccl import get_unique_id  # noqa: F401
major, minor = [int(x) for x in np.__version__.split(".")[:2]]
if (major, minor) >= (2, 3):
    raise SystemExit(f"numpy {np.__version__} is too new for numba")
print(f"numpy_version={np.__version__}")
print("cupy_nccl_available=1")
PY_CUPY_IMPORT
      python3 -P -m pip install --no-cache-dir --only-binary=:all: --upgrade --force-reinstall --target "${PYDEPS_DIR}" "numpy<2.3,>=2.0" "cupy-cuda12x" "nvidia-nccl-cu12>=2.18.1,<3"
    fi
    PYTHONPATH="${PYDEPS_DIR}" python3 -P - <<'PY_CUPY_CHECK'
import numpy as np
import cupy
from cupy.cuda.nccl import get_unique_id
major, minor = [int(x) for x in np.__version__.split(".")[:2]]
if (major, minor) >= (2, 3):
    raise SystemExit(f"numpy {np.__version__} is too new for numba")
print(f"numpy_version={np.__version__}")
print(f"cupy_version={cupy.__version__}")
print("cupy_nccl_import=ok")
PY_CUPY_CHECK
  ) || {
    status=$?
    echo "cupy setup failed with status ${status}" > "${CUPY_FAIL_FILE}"
    exit "${status}"
  }
  echo ready > "${CUPY_READY_FILE}"
else
  for _ in $(seq 1 "${HEAD_WAIT_SECONDS}"); do
    [[ -s "${CUPY_FAIL_FILE}" ]] && break
    [[ -s "${CUPY_READY_FILE}" ]] && break
    sleep 1
  done
  if [[ -s "${CUPY_FAIL_FILE}" ]]; then
    echo "cupy setup failed on rank0"
    cat "${CUPY_FAIL_FILE}"
    exit 1
  fi
  test -s "${CUPY_READY_FILE}"
  PYTHONPATH="${PYDEPS_DIR}" python3 -P - <<'PY_CUPY_CHECK'
import numpy as np
import cupy
from cupy.cuda.nccl import get_unique_id
major, minor = [int(x) for x in np.__version__.split(".")[:2]]
if (major, minor) >= (2, 3):
    raise SystemExit(f"numpy {np.__version__} is too new for numba")
print(f"numpy_version={np.__version__}")
print(f"cupy_version={cupy.__version__}")
print("cupy_nccl_import=ok")
PY_CUPY_CHECK
fi

ray stop --force || true
sleep "${RAY_CLEANUP_SETTLE_SECONDS:-0}"
ray stop --force || true

THIS_IP=$(getent hosts "$(hostname)" | awk '{print $1; exit}' || true)
if [[ -z "${THIS_IP}" ]]; then
  THIS_IP=$(hostname -I | tr ' ' '\n' | awk '$1 !~ /^169[.]254[.]/ {print; exit}' || true)
fi
test -n "${THIS_IP}"
echo "this_ip=${THIS_IP}"

if mkdir "${HEAD_LOCK_DIR}" 2>/dev/null; then
  IS_HEAD=1
else
  IS_HEAD=0
fi
echo "is_head=${IS_HEAD}"
setup_flexkv

if [[ "${IS_HEAD}" != "1" ]]; then
  for _ in $(seq 1 "${HEAD_WAIT_SECONDS}"); do
    [[ -s "${HEAD_FILE}" ]] && break
    sleep 1
  done
  test -s "${HEAD_FILE}"
  HEAD_ADDR=$(cat "${HEAD_FILE}")
  ray start --address="${HEAD_ADDR}" \
    --node-ip-address="${THIS_IP}" \
    --num-gpus="${NGPUS_PER_NODE}"
  for _ in $(seq 1 14400); do
    [[ -e "${DONE_FILE}" ]] && break
    sleep 1
  done
  ray stop --force || true
  echo "END worker $(date -Is)"
  exit 0
fi

rm -f "${HEAD_FILE}" "${DONE_FILE}"
HEAD_ADDR="${THIS_IP}:${RAY_PORT}"
echo "${HEAD_ADDR}" > "${HEAD_FILE}"
echo "head_rank=${RANK}" > "${LOGDIR}/ray_head_${JOB_ID}.meta"

ray start --head \
  --node-ip-address="${THIS_IP}" \
  --port="${RAY_PORT}" \
  --include-dashboard=False \
  --num-gpus="${NGPUS_PER_NODE}" \
  --temp-dir="${RAY_TEMP_DIR}" \
  --disable-usage-stats

MLFLOW_PID=""
cleanup() {
  touch "${DONE_FILE}" || true
  cleanup_flexkv
  if [[ -n "${MLFLOW_PID}" ]]; then
    kill "${MLFLOW_PID}" 2>/dev/null || true
    wait "${MLFLOW_PID}" 2>/dev/null || true
  fi
  ray stop --force || true
}
trap cleanup EXIT

export RAY_ADDRESS="${HEAD_ADDR}"
python3 -P - <<'PY'
import os
import time
import ray

ray.init(address=os.environ["RAY_ADDRESS"])
deadline = time.time() + int(os.environ.get("CLUSTER_WAIT_SECONDS", "900"))
while time.time() < deadline:
    nodes = [n for n in ray.nodes() if n.get("Alive")]
    total_gpus = sum(n.get("Resources", {}).get("GPU", 0) for n in nodes)
    print(f"waiting_ray_nodes={len(nodes)} total_gpus={total_gpus}")
    if len(nodes) >= int(os.environ["NNODES_TOTAL"]) and total_gpus >= int(os.environ["NNODES_TOTAL"]) * int(os.environ["NGPUS_PER_NODE"]):
        break
    time.sleep(5)
nodes = [n for n in ray.nodes() if n.get("Alive")]
total_gpus = sum(n.get("Resources", {}).get("GPU", 0) for n in nodes)
print(f"ray_alive_nodes={len(nodes)}")
print(f"ray_total_gpus={total_gpus}")
for n in nodes:
    print(f"ray_node={n.get('NodeManagerAddress')} resources={n.get('Resources')}")
assert len(nodes) >= int(os.environ["NNODES_TOTAL"]), nodes
assert total_gpus >= int(os.environ["NNODES_TOTAL"]) * int(os.environ["NGPUS_PER_NODE"]), total_gpus
ray.shutdown()
PY

export MLFLOW_TRACKING_URI="http://${THIS_IP}:${MLFLOW_PORT}"
export PYTHONPATH="${RUNTIME_PYTHONPATH}"
test -d "${MLFLOW_SITE}/mlflow"
test -s "${MLFLOW_DUMP_SCRIPT}"
python3 -c 'import numpy, cupy; from cupy.cuda import nccl; import mlflow; assert hasattr(nccl, "get_unique_id"); print("runtime_numpy={} cupy={} mlflow={}".format(numpy.__version__, cupy.__version__, mlflow.__version__))'
export MLFLOW_VERSION
MLFLOW_VERSION=$(python3 -c 'import mlflow; print(mlflow.__version__)')
python3 -m mlflow server \
  --host 0.0.0.0 \
  --port "${MLFLOW_PORT}" \
  --backend-store-uri "sqlite:///${MLFLOW_DB}" \
  --artifacts-destination "${MLFLOW_ARTIFACT_ROOT}" \
  > "${MLFLOW_RUN_DIR}/mlflow-server.log" 2>&1 &
MLFLOW_PID=$!
for _ in $(seq 1 120); do
  if curl -fsS "${MLFLOW_TRACKING_URI}/health" >/dev/null; then
    break
  fi
  sleep 1
done
curl -fsS "${MLFLOW_TRACKING_URI}/health"
echo "mlflow_tracking_uri=${MLFLOW_TRACKING_URI}"
echo "mlflow_project=${MLFLOW_PROJECT}"
echo "mlflow_patch_sha256=${VERL_PATCH_SHA256}"

export PYTHONPATH="${RUNTIME_PYTHONPATH}"
echo "PYTHONPATH=${PYTHONPATH}"

rollout_mode="async"
rollout_name=${ROLLOUT_NAME:-vllm}
return_raw_chat="True"

max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
max_response_length=${MAX_RESPONSE_LENGTH:-512}
max_model_len=${MAX_MODEL_LEN:-$((max_prompt_length + max_response_length))}
actor_ppo_max_token_len=${ACTOR_PPO_MAX_TOKEN_LEN:-$(((max_prompt_length + max_response_length) * 2))}
infer_ppo_max_token_len=${INFER_PPO_MAX_TOKEN_LEN:-$(((max_prompt_length + max_response_length) * 3))}
train_prompt_bsz=0
gen_prompt_bsz=1
n_resp_per_prompt=${N_RESP_PER_PROMPT:-2}
train_prompt_mini_bsz=${TRAIN_PROMPT_MINI_BSZ:-16}
rollout_max_num_seqs=${ROLLOUT_MAX_NUM_SEQS:-2}
rollout_tp=${ROLLOUT_TP:-1}
rollout_enforce_eager=${ROLLOUT_ENFORCE_EAGER:-False}
trainer_tp=${TRAINER_TP:-4}
trainer_pp=${TRAINER_PP:-1}
trainer_ep=${TRAINER_EP:-1}
trainer_etp=${TRAINER_ETP:-1}
trainer_cp=${TRAINER_CP:-1}
trainer_ppo_micro_bsz=${TRAINER_PPO_MICRO_BSZ:-1}
mcore_model_path=${MCORE_MODEL_PATH:-}
if [[ -n "${mcore_model_path}" ]]; then
  use_dist_checkpointing=True
  dist_checkpointing_path="${mcore_model_path}"
else
  use_dist_checkpointing=False
  dist_checkpointing_path=null
fi
total_rollout_steps=${TOTAL_ROLLOUT_STEPS:-64}
staleness_threshold=${STALENESS_THRESHOLD:-1.0}
trigger_parameter_sync_step=${TRIGGER_PARAMETER_SYNC_STEP:-1}
require_batches=${REQUIRE_BATCHES:-1}
dynamic_deactivate_ratio=${DYNAMIC_DEACTIVATE_RATIO:-1.0}
partial_rollout=True
concurrent_samples_per_replica=${CONCURRENT_SAMPLES_PER_REPLICA:-2}
trainer_total_epochs=${TRAINER_TOTAL_EPOCHS:-1}
trainer_save_freq=${TRAINER_SAVE_FREQ:--1}
enable_overlong_buffer=${ENABLE_OVERLONG_BUFFER:-True}
overlong_buffer_len=${OVERLONG_BUFFER_LEN:-$((max_response_length / 2))}
overlong_penalty_factor=${OVERLONG_PENALTY_FACTOR:-1.0}

required_samples=$((train_prompt_mini_bsz * require_batches))
planned_trainer_steps=${PLANNED_TRAINER_STEPS:-1}
minimum_rollout_steps=$((required_samples * trigger_parameter_sync_step * planned_trainer_steps))
recommended_rollout_steps=$((minimum_rollout_steps * 2))
if [[ "${enable_overlong_buffer}" == "True" ]] && (( overlong_buffer_len >= max_response_length )); then
  echo "ERROR: overlong_buffer_len=${overlong_buffer_len} must be smaller than max_response_length=${max_response_length}" >&2
  exit 2
fi
if (( total_rollout_steps < minimum_rollout_steps )); then
  echo "ERROR: total_rollout_steps=${total_rollout_steps} < minimum_rollout_steps=${minimum_rollout_steps}" >&2
  exit 2
fi
echo "dataset_mode=dapo"
echo "dapo_revision=${DAPO_REVISION}"
echo "required_samples=${required_samples}"
echo "minimum_rollout_steps=${minimum_rollout_steps}"
echo "recommended_rollout_steps=${recommended_rollout_steps}"
echo "configured_total_rollout_steps=${total_rollout_steps}"
echo "dynamic_deactivate_ratio=${dynamic_deactivate_ratio}"
echo "rollout_tp=${rollout_tp}"
echo "rollout_enforce_eager=${rollout_enforce_eager}"
echo "trainer_backend=megatron"
echo "trainer_tp=${trainer_tp}"
echo "trainer_pp=${trainer_pp}"
echo "trainer_ep=${trainer_ep}"
echo "trainer_etp=${trainer_etp}"
echo "trainer_cp=${trainer_cp}"
echo "use_dist_checkpointing=${use_dist_checkpointing}"
echo "dist_checkpointing_path=${dist_checkpointing_path}"
echo "max_model_len=${max_model_len}"


python3 -m verl.experimental.fully_async_policy.fully_async_main \
    --config-path=config \
    --config-name=fully_async_ppo_megatron_trainer.yaml \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.return_raw_chat=${return_raw_chat} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=1 \
    actor_rollout_ref.actor.optim.lr_decay_steps=${total_rollout_steps} \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${trainer_ppo_micro_bsz} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${trainer_tp} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${trainer_pp} \
    actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=null \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${trainer_ep} \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${trainer_etp} \
    actor_rollout_ref.actor.megatron.context_parallel_size=${trainer_cp} \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.masked_softmax_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_activation_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_dropout_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.deallocate_pipeline_outputs=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.persist_layer_norm=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True \
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.megatron.grad_offload=True \
    actor_rollout_ref.actor.megatron.optimizer_offload=True \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=${use_dist_checkpointing} \
    actor_rollout_ref.actor.megatron.dist_checkpointing_path="${dist_checkpointing_path}" \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.35} \
    actor_rollout_ref.rollout.standalone_gpu_memory_utilization=${STANDALONE_GPU_MEMORY_UTILIZATION:-0.35} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp} \
    actor_rollout_ref.rollout.enforce_eager=${rollout_enforce_eager} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_model_len=${max_model_len} \
    actor_rollout_ref.rollout.ignore_eos=${ROLLOUT_IGNORE_EOS} \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.max_num_seqs=${rollout_max_num_seqs} \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.trace.backend=mlflow \
    actor_rollout_ref.rollout.trace.project_name="${MLFLOW_PROJECT}" \
    actor_rollout_ref.rollout.trace.experiment_name="${exp_name}" \
    actor_rollout_ref.rollout.trace.token2text=False \
    actor_rollout_ref.rollout.trace.max_samples_per_step_per_worker=null \
    "${KV_ARGS[@]}" \
    reward.reward_manager.name=dapo \
    +reward.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.val_before_train=False \
    trainer.save_freq="${trainer_save_freq}" \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=disable \
    trainer.nnodes="${NNODES_TRAIN}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.total_epochs="${trainer_total_epochs}" \
    trainer.test_freq=-1 \
    trainer.log_val_generations=0 \
    rollout.nnodes="${NNODES_ROLLOUT}" \
    rollout.n_gpus_per_node="${NGPUS_PER_NODE}" \
    rollout.total_rollout_steps="${total_rollout_steps}" \
    async_training.staleness_threshold="${staleness_threshold}" \
    async_training.trigger_parameter_sync_step="${trigger_parameter_sync_step}" \
    async_training.require_batches="${require_batches}" \
    async_training.partial_rollout="${partial_rollout}" \
    async_training.use_trainer_do_validate=False \
    async_training.use_dynamic_resource_scaling=True \
    async_training.dynamic_scaling_deactivate_ratio="${dynamic_deactivate_ratio}" \
    +async_training.concurrent_samples_per_replica="${concurrent_samples_per_replica}" \
    +ray_kwargs.ray_init.address="${HEAD_ADDR}" \
    +ray_kwargs.ray_init._temp_dir="${RAY_TEMP_DIR}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_RPC_BASE_PATH="${VLLM_RPC_BASE_PATH}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.TMPDIR="${TMPDIR}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.TORCH_CUDA_ARCH_LIST="'${TORCH_CUDA_ARCH_LIST}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH="${PYTHONPATH}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_ALLOW_LONG_MAX_MODEL_LEN="'1'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_USE_V1="'1'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.VERL_RECOMPUTE_TRACE="'1'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.VERL_RECOMPUTE_DEACTIVATE_GRACE_S="'${VERL_RECOMPUTE_DEACTIVATE_GRACE_S}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.MLFLOW_ENABLE_ASYNC_TRACE_LOGGING="'${MLFLOW_ENABLE_ASYNC_TRACE_LOGGING}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.RAY_DEDUP_LOGS="'0'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.WANDB_ENTITY="${WANDB_ENTITY}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.WANDB_PROJECT="${WANDB_PROJECT}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.WANDB_MODE="${WANDB_MODE}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.WANDB_RUN_ID="${WANDB_RUN_ID}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.WANDB_NAME="${WANDB_NAME}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.WANDB_RESUME="${WANDB_RESUME}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.WANDB_GROUP="${WANDB_GROUP}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.WANDB_TAGS="'${WANDB_TAGS}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.WANDB_DIR="${WANDB_DIR}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.WANDB_CACHE_DIR="${WANDB_CACHE_DIR}"

python3 "${MLFLOW_DUMP_SCRIPT}" \
  --output-dir "${MLFLOW_OUTPUT_DIR}" \
  --project-name "${MLFLOW_PROJECT}" \
  --expected-logical-requests "${EXPECTED_LOGICAL_REQUESTS}" \
  --expected-server-replicas $(((NNODES_ROLLOUT + NNODES_TRAIN) * NGPUS_PER_NODE / ROLLOUT_TP)) \
  --minimum-retry-attempts "${MINIMUM_RETRY_ATTEMPTS}"

echo "wandb_run_id=${WANDB_RUN_ID}"
echo "END $(date -Is)"
