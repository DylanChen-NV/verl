#!/usr/bin/env bash
set -euo pipefail
export ABC_GROUP=C
export VERL_RUN_ID=mcfd1
export MLFLOW_PROJECT=dynamic-recompute-a80-moon-c-final-dispatch-r1
export MLFLOW_PORT=15020
export MOONCAKE_MASTER_PORT=15030
export MOONCAKE_METRICS_PORT=15031
exec /lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/05_claude_ws/dynamic_mooncake_abort_dev/scripts/dfw_h100_dynamic_recompute_30b_r8k_req32_mooncake_abc.sh
