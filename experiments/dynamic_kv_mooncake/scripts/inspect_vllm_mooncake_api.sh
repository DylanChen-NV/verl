#!/usr/bin/env bash
set -euo pipefail

python3 - <<'PY'
import inspect

import vllm
from vllm.v1.engine.core import EngineCore
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.connector import (
    MooncakeStoreConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.scheduler import (
    MooncakeStoreScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker import (
    MooncakeStoreWorker,
)

print(f"vllm_version={vllm.__version__}")
for cls, methods in (
    (EngineCore, ("sleep", "_reset_caches", "abort_requests")),
    (Scheduler, ("finish_requests", "_free_request")),
    (MooncakeStoreScheduler, ("request_finished", "get_finished")),
    (MooncakeStoreConnector, ("request_finished", "get_finished", "reset_cache")),
    (MooncakeStoreWorker, ("get_finished",)),
):
    print(f"CLASS {cls.__module__}.{cls.__name__}")
    print(f"FILE {inspect.getsourcefile(cls)}")
    for method_name in methods:
        method = getattr(cls, method_name, None)
        if method is None:
            print(f"MISSING {method_name}")
            continue
        print(f"METHOD {method_name}{inspect.signature(method)}")
        print(inspect.getsource(method))

print("FULL_CLASS MooncakeStoreScheduler")
print(inspect.getsource(MooncakeStoreScheduler))
PY
