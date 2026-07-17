#!/usr/bin/env bash
set -euo pipefail
python3 -S - <<PY
from pathlib import Path

path = Path("/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/05_claude_ws/dynamic_mooncake_abort_dev/verl_repo/verl/workers/rollout/vllm_rollout/vllm_async_server.py")
text = path.read_text()
marker = "pause_generation() does not return until vLLM delayed-free"
if marker not in text:
    func = text.index("    async def abort_all_requests(")
    pause_start = text.index("                await self.engine.pause_generation(", func)
    pause_end = text.index("                )\n", pause_start) + len("                )\n")
    barrier_start = text.index("                if checkpoint_kv:\n", pause_end)
    wait_start = text.index("                    await self.engine.wait_for_requests_to_drain(", barrier_start)
    wait_end = text.index("                    )\n", wait_start) + len("                    )\n")
    pause = text[pause_start:pause_end]
    event = text[barrier_start:wait_start]
    replacement = (
        event
        + "                # pause_generation() does not return until vLLM delayed-free\n"
        + "                # connector work reaches a terminal state.\n"
        + pause
        + "                if checkpoint_kv:\n"
    )
    path.write_text(text[:pause_start] + replacement + text[wait_end:])
    print("prepared_repo_abort_barrier_patch=True")
else:
    print("prepared_repo_abort_barrier_patch=False")

path = Path("/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/05_claude_ws/dynamic_mooncake_abort_dev/verl_repo/verl/workers/rollout/llm_server.py")
text = path.read_text()
old = """                request_id=(
                    request_id
                    if "__verl_recompute_attempt_" in request_id
                    else uuid4().hex
                ),"""
new = "                request_id=request_id,"
if old in text:
    path.write_text(text.replace(old, new, 1))
    print("prepared_repo_logical_request_id_patch=True")
elif new in text:
    print("prepared_repo_logical_request_id_patch=False")
else:
    raise SystemExit("prepared repo logical request ID target not found")
PY
