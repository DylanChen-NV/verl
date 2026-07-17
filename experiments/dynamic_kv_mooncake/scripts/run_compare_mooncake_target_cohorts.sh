#!/usr/bin/env bash
set -euo pipefail

ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/05_claude_ws/dynamic_mooncake_abort_dev
gzip -dc "${ROOT}/scripts/compare_mooncake_target_cohorts.py.gz" > "${ROOT}/scripts/compare_mooncake_target_cohorts.py"
python3 "${ROOT}/scripts/compare_mooncake_target_cohorts.py" \
  "${ROOT}/monitoring/runs" \
  --json-out "${ROOT}/monitoring/mooncake_abc_target_comparison.json"
