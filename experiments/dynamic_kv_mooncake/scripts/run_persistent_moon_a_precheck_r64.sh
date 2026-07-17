#!/usr/bin/env bash
set -euo pipefail

export ABC_GROUP=A
export VERL_RUN_ID=mapre64
export MLFLOW_PROJECT=dynamic-recompute-moon-a-precheck-r64
export MLFLOW_PORT=15070
export PLANNED_TRAINER_STEPS=4
export TOTAL_ROLLOUT_STEPS=64
export EXPECTED_MLFLOW_TRACES=128
export MINIMUM_RETRY_ATTEMPTS=1
export DYNAMIC_DEACTIVATE_RATIO=0.0625
exec /lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/05_claude_ws/dynamic_mooncake_abort_dev/scripts/dfw_h100_dynamic_recompute_30b_r8k_req32_mooncake_abc.sh
