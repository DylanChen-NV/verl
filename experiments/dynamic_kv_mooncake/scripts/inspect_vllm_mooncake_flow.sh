#!/usr/bin/env bash
set -euo pipefail

python3 - <<'PY'
import inspect

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import data
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.connector import MooncakeStoreConnector
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.scheduler import MooncakeStoreScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine.core import EngineCore

targets = (
    (Scheduler, "_connector_finished"),
    (Scheduler, "schedule"),
    (EngineCore, "pause_scheduler"),
    (EngineCore, "step_with_batch_queue"),
    (MooncakeStoreConnector, "build_connector_meta"),
    (MooncakeStoreConnector, "request_finished_all_groups"),
)
for cls, name in targets:
    method = getattr(cls, name, None)
    print(f"TARGET {cls.__module__}.{cls.__name__}.{name}")
    if method is None:
        print("MISSING")
    else:
        print(inspect.getsource(method))

for cls_name in ("RequestTracker", "ReqMeta", "MooncakeStoreConnectorMetadata"):
    cls = getattr(data, cls_name)
    print(f"CLASS {cls.__module__}.{cls.__name__}")
    print(inspect.getsource(cls))
PY
