#!/usr/bin/env python3
from pathlib import Path


path = Path(
    "/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/"
    "05_claude_ws/dynamic_mooncake_abort_dev/scripts/"
    "dfw_h100_dynamic_recompute_30b_r8k_req32_mooncake_abc.sh"
)
text = path.read_text()
if "export MINIMUM_RETRY_ATTEMPTS=" not in text:
    text = text.replace(
        "export EXPECTED_MLFLOW_TRACES=${EXPECTED_MLFLOW_TRACES:-120}\n",
        "export EXPECTED_MLFLOW_TRACES=${EXPECTED_MLFLOW_TRACES:-120}\n"
        "export MINIMUM_RETRY_ATTEMPTS=${MINIMUM_RETRY_ATTEMPTS:-16}\n",
        1,
    )
text = text.replace(
    "--minimum-retry-attempts 16",
    '--minimum-retry-attempts "${MINIMUM_RETRY_ATTEMPTS}"',
    1,
)
path.write_text(text)
print("minimum_retry_parameterized=PASS")
