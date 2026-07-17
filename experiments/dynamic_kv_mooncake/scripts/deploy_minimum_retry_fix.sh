#!/usr/bin/env bash
set -euo pipefail

root=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/05_claude_ws/dynamic_mooncake_abort_dev/scripts
python3 "${root}/patch_remote_minimum_retry.py"
if [[ -f "${root}/run_persistent_moon_c_cross_node_precheck.sh.gz" ]]; then
  gzip -dc "${root}/run_persistent_moon_c_cross_node_precheck.sh.gz" > "${root}/run_persistent_moon_c_cross_node_precheck.sh"
  chmod +x "${root}/run_persistent_moon_c_cross_node_precheck.sh"
fi
grep -nE 'MINIMUM_RETRY|minimum-retry' \
  "${root}/dfw_h100_dynamic_recompute_30b_r8k_req32_mooncake_abc.sh" \
  "${root}/run_persistent_moon_c_cross_node_precheck.sh"
