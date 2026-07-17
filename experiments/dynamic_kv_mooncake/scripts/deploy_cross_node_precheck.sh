#!/usr/bin/env bash
set -euo pipefail
root=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/05_claude_ws/dynamic_mooncake_abort_dev/scripts
for name in \
  apply_vllm_mooncake_abort_reuse.py \
  dfw_h100_dynamic_recompute_30b_r8k_req32_mooncake_abc.sh \
  patch_prepared_repo_abort_barrier.py.sh \
  run_persistent_moon_c_cross_node_precheck.sh
do
  gzip -dc "${root}/${name}.gz" > "${root}/${name}"
done
chmod +x "${root}"/*.sh "${root}"/*.py
"${root}/patch_prepared_repo_abort_barrier.py.sh"
