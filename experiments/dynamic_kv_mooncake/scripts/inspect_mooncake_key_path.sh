#!/usr/bin/env bash
set -euo pipefail

python3 - <<'PY'
import inspect

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import connector, data, scheduler, worker

targets = (
    (scheduler.MooncakeStoreScheduler, "get_num_new_matched_tokens"),
    (scheduler.MooncakeStoreScheduler, "request_finished"),
    (connector.MooncakeStoreConnector, "get_num_new_matched_tokens"),
    (worker.MooncakeStoreWorker, "lookup"),
    (worker.MooncakeStoreWorker, "get_finished"),
)
for cls, name in targets:
    print(f"TARGET {cls.__module__}.{cls.__name__}.{name}")
    method = getattr(cls, name, None)
    print("MISSING" if method is None else inspect.getsource(method))

for module in (scheduler, worker):
    for name, value in sorted(vars(module).items()):
        if callable(value) and any(word in name.lower() for word in ("key", "hash")):
            print(f"FUNCTION {module.__name__}.{name}")
            try:
                print(inspect.getsource(value))
            except (OSError, TypeError):
                pass

for cls_name in ("ReqMeta", "RequestTracker"):
    cls = getattr(data, cls_name)
    print(f"CLASS {cls.__module__}.{cls.__name__}")
    print(inspect.getsource(cls))
PY
