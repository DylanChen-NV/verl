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
"""Rollout performance tracing utilities."""

from verl.utils.rollout_perf.collector import (
    emit_counter,
    emit_event,
    get_trace_context_for_rpc,
    init_rollout_perf,
    is_current_trace_enabled,
    is_rollout_perf_enabled,
    start_span,
)
from verl.utils.rollout_perf.config import PerfTraceConfig, PerfTraceEngineInternalConfig
from verl.utils.rollout_perf.context import current_trace_context, push_trace_context

__all__ = [
    "PerfTraceConfig",
    "PerfTraceEngineInternalConfig",
    "current_trace_context",
    "emit_counter",
    "emit_event",
    "get_trace_context_for_rpc",
    "init_rollout_perf",
    "is_current_trace_enabled",
    "is_rollout_perf_enabled",
    "push_trace_context",
    "start_span",
]
