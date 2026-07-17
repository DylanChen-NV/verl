#!/usr/bin/env bash
set -euo pipefail
export ABC_GROUP=C
export VERL_RUN_ID=mcpre1
export MLFLOW_PROJECT=dynamic-recompute-moon-c-cross-node-precheck
export MLFLOW_PORT=15040
export MOONCAKE_MASTER_PORT=15050
export MOONCAKE_METRICS_PORT=15051
export PLANNED_TRAINER_STEPS=1
export TOTAL_ROLLOUT_STEPS=16
export EXPECTED_MLFLOW_TRACES=32
export MINIMUM_RETRY_ATTEMPTS=1
export DYNAMIC_DEACTIVATE_RATIO=0.0625
exec /lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/05_claude_ws/dynamic_mooncake_abort_dev/scripts/dfw_h100_dynamic_recompute_30b_r8k_req32_mooncake_abc.sh
