#!/usr/bin/env bash
set -euo pipefail

ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/05_claude_ws
WORK=${ROOT}/dynamic_mooncake_abort_dev
SESSION=${ROOT}/.gpu-workflow/dfw/persist/dyn-moon-final-dispatch-20260717
TOOLS=${WORK}/scripts
RUNS=${WORK}/monitoring/runs
cat "${TOOLS}"/export_dynamic_recompute_grafana.py.gz.part-* > "${TOOLS}/export_dynamic_recompute_grafana.py.gz"

for tool in join_dynamic_recompute_trajectory export_dynamic_recompute_grafana validate_dynamic_recompute_grafana; do
  gzip -dc "${TOOLS}/${tool}.py.gz" > "${TOOLS}/${tool}.py"
done

while read -r run_id command_id; do
  run=${RUNS}/${run_id}
  joined=${run}/joined
  grafana=${run}/grafana
  raw_log=${SESSION}/logs/${command_id}.rank-0.out
  mkdir -p "${joined}" "${grafana}"
  python3 "${TOOLS}/join_dynamic_recompute_trajectory.py" \
    --spans-csv "${run}/trace_dump/spans.csv" \
    --strict-report "${run}/analysis/queue_excluded_report.json" \
    --raw-log "${raw_log}" \
    --output-dir "${joined}"
  python3 "${TOOLS}/export_dynamic_recompute_grafana.py" \
    --spans-csv "${run}/trace_dump/spans.csv" \
    --joined-csv "${joined}/trajectory_recompute_attempts.csv" \
    --joined-summary "${joined}/trajectory_recompute_summary.json" \
    --strict-report "${run}/analysis/queue_excluded_report.json" \
    --raw-log "${raw_log}" \
    --output-dir "${grafana}" \
    --run-id "${run_id}"
  python3 "${TOOLS}/validate_dynamic_recompute_grafana.py" --output-dir "${grafana}"
done <<'RUNS'
ma80r1 cmd-20260717-093957-depo
mb80r1 cmd-20260717-095904-tnb4
mc80r2 cmd-20260717-103928-2mz4
RUNS
