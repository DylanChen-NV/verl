#!/usr/bin/env bash
set -euo pipefail

ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/05_claude_ws/dynamic_mooncake_abort_dev
OUT=${ROOT}/monitoring/runs/mc80r2/analysis
gzip -dc "${ROOT}/scripts/summarize_mooncake_proven_reuse.py.gz" > "${ROOT}/scripts/summarize_mooncake_proven_reuse.py"
python3 "${ROOT}/scripts/summarize_mooncake_proven_reuse.py" \
  "${OUT}/strict_cross_node_reuse_report.json" \
  "${OUT}/queue_excluded_report.json" \
  --json-out "${OUT}/proven_reuse_summary.json"
