#!/usr/bin/env bash
set -euo pipefail

python3 - <<'PY'
import inspect

from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core import EngineCore
from vllm.v1.core.sched.scheduler import Scheduler

for cls, names in (
    (AsyncLLM, ("pause_generation", "sleep", "reset_prefix_cache")),
    (EngineCore, ("step", "has_unfinished_requests")),
    (Scheduler, ("has_requests", "get_num_unfinished_requests")),
):
    for name in names:
        method = getattr(cls, name, None)
        print(f"TARGET {cls.__module__}.{cls.__name__}.{name}")
        if method is None:
            print("MISSING")
        else:
            print(inspect.getsource(method))
PY
