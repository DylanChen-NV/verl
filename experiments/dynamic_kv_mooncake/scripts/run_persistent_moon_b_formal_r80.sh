#!/usr/bin/env bash
set -euo pipefail
export ABC_GROUP=B VERL_RUN_ID=mb80r1
export MLFLOW_PROJECT=dynamic-recompute-moon-b-formal-r80
export MLFLOW_PORT=15110 MOONCAKE_MASTER_PORT=15111 MOONCAKE_METRICS_PORT=15112
export PLANNED_TRAINER_STEPS=5 TOTAL_ROLLOUT_STEPS=80
export EXPECTED_MLFLOW_TRACES=160 MINIMUM_RETRY_ATTEMPTS=16
export DYNAMIC_DEACTIVATE_RATIO=0.0625
exec /lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/05_claude_ws/dynamic_mooncake_abort_dev/scripts/dfw_h100_dynamic_recompute_30b_r8k_req32_mooncake_abc.sh
