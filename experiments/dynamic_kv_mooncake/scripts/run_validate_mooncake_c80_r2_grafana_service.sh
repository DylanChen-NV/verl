#!/usr/bin/env bash
set -euo pipefail

ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/05_claude_ws
DASH=${ROOT}/dynamic_mooncake_abort_dev/monitoring/runs/mc80r2/grafana
PORT=31598
DASHBOARD_DIR=${DASH} GRAFANA_PORT=${PORT} \
  bash ${ROOT}/dynamic_recompute_h100/monitoring/tools/start_dynamic_recompute_grafana.sh
python3 ${ROOT}/dynamic_recompute_h100/monitoring/tools/validate_dynamic_recompute_grafana_service.py \
  --base-url http://127.0.0.1:${PORT} \
  --dashboard-uid verl-dyn-recompute-mc80r2 \
  --output-dir ${DASH}
