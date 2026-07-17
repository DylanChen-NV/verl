#!/usr/bin/env bash
set -euo pipefail

root=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/05_claude_ws
run_dir=${root}/dynamic_mooncake_abort_dev/monitoring/runs/mcpre1
mlflow_site=${root}/exp_verl_main_mlflow_perfetto/deps/mlflow-3.14.0
dump_script=${root}/dynamic_recompute_h100/monitoring/dump_dynamic_recompute_mlflow.py
port=15060

export PYTHONPATH="${root}/dynamic_recompute_h100/pydeps_cupy_npy22:${root}/dynamic_mooncake_abort_dev/verl_repo:${mlflow_site}"
export MLFLOW_TRACKING_URI="http://127.0.0.1:${port}"
python3 -m mlflow server \
  --host 127.0.0.1 \
  --port "${port}" \
  --backend-store-uri "sqlite:///${run_dir}/mlflow.db" \
  --artifacts-destination "${run_dir}/artifacts" \
  > "${run_dir}/mlflow-redump-server.log" 2>&1 &
pid=$!
cleanup() {
  kill "${pid}" 2>/dev/null || true
  wait "${pid}" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 120); do
  if curl -fsS "${MLFLOW_TRACKING_URI}/health" >/dev/null; then
    break
  fi
  sleep 1
done
curl -fsS "${MLFLOW_TRACKING_URI}/health"
python3 "${dump_script}" \
  --output-dir "${run_dir}/trace_dump" \
  --project-name dynamic-recompute-moon-c-cross-node-precheck \
  --expected-traces 32 \
  --expected-server-replicas 8 \
  --minimum-retry-attempts 1
