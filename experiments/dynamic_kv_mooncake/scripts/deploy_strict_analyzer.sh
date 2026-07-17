#!/usr/bin/env bash
set -euo pipefail
root=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/05_claude_ws/dynamic_mooncake_abort_dev/scripts
gzip -dc "${root}/analyze_mooncake_strict_reuse.py.gz" > "${root}/analyze_mooncake_strict_reuse.py"
chmod +x "${root}/analyze_mooncake_strict_reuse.py"
