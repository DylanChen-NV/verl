#!/usr/bin/env bash
set -euo pipefail
export ABC_GROUP=C VERL_RUN_ID=mc80r1
export MLFLOW_PROJECT=dynamic-recompute-moon-c-formal-r80
export MLFLOW_PORT=15120 MOONCAKE_MASTER_PORT=15121 MOONCAKE_METRICS_PORT=15122
export PLANNED_TRAINER_STEPS=5 TOTAL_ROLLOUT_STEPS=80
export EXPECTED_MLFLOW_TRACES=160 MINIMUM_RETRY_ATTEMPTS=16
export DYNAMIC_DEACTIVATE_RATIO=0.0625
exec bash /lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/05_claude_ws/dynamic_mooncake_abort_dev/scripts/postprocess_mooncake_formal_run.sh mc80r2 cmd-20260717-103928-2mz4 C
