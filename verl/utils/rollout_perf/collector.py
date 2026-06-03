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
import time
import traceback
from typing import Any, Optional

from verl.utils.rollout_perf.context import current_trace_context
from verl.utils.rollout_perf.writer import init_writer, write_record, writer_enabled


def init_rollout_perf(config: Any, role: str):
    """Initialize rollout perf tracing in the current process."""
    return init_writer(config, role)


def is_rollout_perf_enabled() -> bool:
    return writer_enabled()


def is_current_trace_enabled() -> bool:
    if not writer_enabled():
        return False
    context = current_trace_context()
    return context.get("trace_sampled", True) is not False


def get_trace_context_for_rpc() -> Optional[dict[str, Any]]:
    """Return a serializable trace context for Ray RPC propagation."""
    if not is_current_trace_enabled():
        return None
    return current_trace_context()


def emit_event(name: str, payload: Optional[dict[str, Any]] = None, *, priority: str = "P0") -> None:
    if not is_current_trace_enabled():
        return
    write_record(
        {
            "record_type": "event",
            "name": name,
            "priority": priority,
            "context": current_trace_context(),
            "payload": payload or {},
        }
    )


def emit_counter(name: str, value: float, payload: Optional[dict[str, Any]] = None, *, priority: str = "P0") -> None:
    if not is_current_trace_enabled():
        return
    write_record(
        {
            "record_type": "counter",
            "name": name,
            "priority": priority,
            "context": current_trace_context(),
            "payload": {"value": value, **(payload or {})},
        }
    )


def emit_unsampled_event(
    name: str,
    payload: Optional[dict[str, Any]] = None,
    *,
    priority: str = "P0",
    context: Optional[dict[str, Any]] = None,
) -> None:
    """Emit an event whenever the process-local writer is enabled.

    This is for engine/process-level telemetry that should not be tied to
    trajectory sampling decisions.
    """
    if not writer_enabled():
        return
    write_record(
        {
            "record_type": "event",
            "name": name,
            "priority": priority,
            "context": context if context is not None else current_trace_context(),
            "payload": payload or {},
        }
    )


def emit_unsampled_counter(
    name: str,
    value: float,
    payload: Optional[dict[str, Any]] = None,
    *,
    priority: str = "P0",
    context: Optional[dict[str, Any]] = None,
) -> None:
    """Emit a counter whenever the process-local writer is enabled.

    Unlike emit_counter(), this intentionally bypasses trace_sampled so
    engine-level counters can represent the full engine workload.
    """
    if not writer_enabled():
        return
    write_record(
        {
            "record_type": "counter",
            "name": name,
            "priority": priority,
            "context": context if context is not None else current_trace_context(),
            "payload": {"value": value, **(payload or {})},
        }
    )


class TraceSpan:
    """A complete-event span that writes one JSONL record when finished."""

    def __init__(self, name: str, payload: Optional[dict[str, Any]] = None, *, priority: str = "P0"):
        self.name = name
        self.priority = priority
        self.context = current_trace_context()
        self.payload = payload or {}
        self.enabled = is_current_trace_enabled()
        self.start_unix_ns = time.time_ns()
        self.start_monotonic_ns = time.monotonic_ns()
        self._finished = False

    def finish(
        self,
        payload: Optional[dict[str, Any]] = None,
        *,
        status: str = "ok",
        error: Optional[BaseException] = None,
    ) -> None:
        if self._finished:
            return
        self._finished = True
        if not self.enabled:
            return
        end_unix_ns = time.time_ns()
        end_monotonic_ns = time.monotonic_ns()
        merged_payload = dict(self.payload)
        if payload:
            merged_payload.update(payload)
        if error is not None:
            merged_payload["error_type"] = type(error).__name__
            merged_payload["error_message"] = str(error)
            merged_payload["error_traceback"] = "".join(traceback.format_exception_only(type(error), error)).strip()
        write_record(
            {
                "record_type": "span",
                "name": self.name,
                "priority": self.priority,
                "context": self.context,
                "payload": merged_payload,
                "status": status,
                "start_unix_ns": self.start_unix_ns,
                "end_unix_ns": end_unix_ns,
                "start_monotonic_ns": self.start_monotonic_ns,
                "end_monotonic_ns": end_monotonic_ns,
                "duration_ns": end_monotonic_ns - self.start_monotonic_ns,
            }
        )

    def __enter__(self) -> "TraceSpan":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is None:
            self.finish()
        else:
            self.finish(status="error", error=exc)
        return False


def start_span(name: str, payload: Optional[dict[str, Any]] = None, *, priority: str = "P0") -> TraceSpan:
    return TraceSpan(name, payload=payload, priority=priority)
