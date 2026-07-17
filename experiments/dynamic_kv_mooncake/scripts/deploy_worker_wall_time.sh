#!/usr/bin/env bash
set -euo pipefail

root=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/05_claude_ws/dynamic_mooncake_abort_dev/scripts
gzip -dc "${root}/apply_mooncake_strict_worker_trace.py.gz" > "${root}/apply_mooncake_strict_worker_trace.py"
python3 "${root}/apply_mooncake_strict_worker_trace.py" --root /vllm
