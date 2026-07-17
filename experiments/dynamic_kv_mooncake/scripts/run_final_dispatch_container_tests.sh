#!/usr/bin/env bash
set -euo pipefail
TEST_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/05_claude_ws/dynamic_mooncake_abort_dev
gzip -dc "${TEST_ROOT}/scripts/apply_vllm_mooncake_abort_reuse.py.gz" > "${TEST_ROOT}/scripts/apply_vllm_mooncake_abort_reuse.py"
python3 "${TEST_ROOT}/scripts/apply_vllm_mooncake_abort_reuse.py" --root /vllm
python3 "${TEST_ROOT}/scripts/apply_vllm_mooncake_abort_reuse.py" --root /vllm
python3 "${TEST_ROOT}/scripts/apply_mooncake_strict_worker_trace.py" --root /vllm
PYTHONPATH=/vllm python3 -m pytest -q "${TEST_ROOT}/tests/test_aborted_final_save.py" "${TEST_ROOT}/tests/test_patch_signatures.py"
PYTHONPATH=/vllm python3 - <<PY
import inspect
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker import MooncakeStoreWorker
from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector

get_finished = inspect.getsource(MooncakeStoreWorker.get_finished)
no_forward = inspect.getsource(ActiveKVConnector.no_forward)
assert "self.kv_send_thread.add_request(request)" in get_finished
assert "self.pre_forward(scheduler_output)" in no_forward
assert "self.post_forward(finished_req_ids, wait_for_save=False)" in no_forward
print("finished_only_dispatch_chain=PASS")
PY
