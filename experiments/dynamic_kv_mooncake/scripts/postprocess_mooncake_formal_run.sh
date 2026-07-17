#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 RUN_ID COMMAND_ID GROUP" >&2
  exit 2
fi

RUN_ID=$1
COMMAND_ID=$2
GROUP=$3
ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/05_claude_ws
SESSION_ROOT=${ROOT}/.gpu-workflow/dfw/persist/dyn-moon-final-dispatch-20260717
RUN_ROOT=${ROOT}/dynamic_mooncake_abort_dev/monitoring/runs/${RUN_ID}
OUT=${RUN_ROOT}/analysis
LOG0=${SESSION_ROOT}/logs/${COMMAND_ID}.rank-0.out
LOG1=${SESSION_ROOT}/logs/${COMMAND_ID}.rank-1.out
TOOLS=${ROOT}/dynamic_mooncake_abort_dev/scripts

mkdir -p "${OUT}"
gzip -dc "${TOOLS}/queue_excluded_interval_report.py.gz" > "${TOOLS}/queue_excluded_interval_report.py"
python3 "${TOOLS}/queue_excluded_interval_report.py" \
  "${LOG0}" \
  --json-out "${OUT}/queue_excluded_report.json" \
  >"${OUT}/queue_excluded_report.txt"

if [[ "${GROUP}" != "A" ]]; then
python3 "${TOOLS}/analyze_mooncake_strict_reuse.py" \
  "${LOG0}" "${LOG1}" \
  --json-out "${OUT}/strict_cross_node_reuse_report.json" \
  --csv-out "${OUT}/strict_cross_node_reuse_requests.csv" \
  >"${OUT}/strict_cross_node_reuse_report.txt"
fi

python3 - "${OUT}/queue_excluded_report.json" "${OUT}/formal_run_summary.json" "${GROUP}" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
cycles = []
for cycle in report.get("cycles", []):
    node_union_ms = sum(node["queue_excluded"]["union_ms"] for node in cycle.get("nodes", []))
    cycles.append({
        "dynamic_cycle_id": cycle.get("dynamic_cycle_id"),
        "status_counts": cycle.get("status_counts", {}),
        "queue_excluded_union_ms_sum_across_nodes": node_union_ms,
        "nodes": [
            {
                "node_id": node.get("node_id"),
                "completed_attempts": node.get("completed_attempts"),
                "queue_union_ms": node["queue"]["union_ms"],
                "queue_excluded_union_ms": node["queue_excluded"]["union_ms"],
            }
            for node in cycle.get("nodes", [])
        ],
    })
summary = {
    "group": sys.argv[3],
    "event_count": report.get("event_count"),
    "attempt_count": report.get("attempt_count"),
    "status_counts": report.get("status_counts", {}),
    "cycles": cycles,
}
Path(sys.argv[2]).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

echo "POSTPROCESS_COMPLETE group=${GROUP} run_id=${RUN_ID} output=${OUT}"
