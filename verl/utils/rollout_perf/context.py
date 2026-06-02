# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

_TRACE_CONTEXT: ContextVar[dict[str, Any]] = ContextVar("rollout_perf_trace_context", default={})


def current_trace_context() -> dict[str, Any]:
    """Return a copy of the current rollout perf trace context."""
    return dict(_TRACE_CONTEXT.get() or {})


@contextmanager
def push_trace_context(**kwargs: Any) -> Iterator[dict[str, Any]]:
    """Temporarily merge rollout perf trace context values."""
    previous = current_trace_context()
    updated = dict(previous)
    updated.update({key: value for key, value in kwargs.items() if value is not None})
    token = _TRACE_CONTEXT.set(updated)
    try:
        yield updated
    finally:
        _TRACE_CONTEXT.reset(token)
