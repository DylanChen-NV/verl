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
"""vLLM stat logger adapter for rollout performance tracing.

The adapter uses vLLM's custom StatLogger hook, which is intentionally less
invasive than patching vLLM scheduler or cache-manager internals. vLLM marks
SchedulerStats/IterationStats as unstable, so this module uses defensive
attribute access and silently skips fields that are unavailable.
"""

import math
import os
import threading
import time
from typing import Any, Optional

from verl.utils.rollout_perf.collector import emit_unsampled_counter, emit_unsampled_event, init_rollout_perf

try:  # vLLM is only required in rollout server processes.
    import vllm
    from vllm.v1.metrics.loggers import StatLoggerBase
except Exception:  # pragma: no cover - allows importing verl without vLLM installed.
    vllm = None
    StatLoggerBase = object


def _getattr(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    return getattr(obj, name, default)


def _as_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception:
            return None
    try:
        return float(value)
    except Exception:
        return None


def _as_int(value: Any) -> Optional[int]:
    number = _as_number(value)
    if number is None:
        return None
    return int(number)


def _safe_len(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return len(value)
    except Exception:
        return None


def _hit_rate(stats: Any) -> Optional[float]:
    rate = _as_number(_getattr(stats, "hit_rate", None))
    if rate is not None:
        return rate
    queries = _as_number(_getattr(stats, "queries", None))
    hits = _as_number(_getattr(stats, "hits", None))
    if queries and hits is not None:
        return hits / queries
    aggregated_queries = _as_number(_getattr(stats, "aggregated_query_total", None))
    aggregated_hits = _as_number(_getattr(stats, "aggregated_query_hit", None))
    if aggregated_queries and aggregated_hits is not None:
        return aggregated_hits / aggregated_queries
    return None


def _config_get(config: Any, name: str, default: Any = None) -> Any:
    if config is None:
        return default
    if hasattr(config, "get"):
        try:
            return config.get(name, default)
        except Exception:
            pass
    return getattr(config, name, default)


def _positive_int(value: Any, default: int) -> int:
    try:
        result = int(value)
    except Exception:
        result = default
    return result if result > 0 else default


def _nearest_rank_percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(math.ceil((percentile / 100.0) * len(ordered))) - 1
    index = max(0, min(index, len(ordered) - 1))
    return float(ordered[index])


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


class _RolloutPerfVLLMActivity:
    """Process-local active rollout request tracker for engine counters."""

    def __init__(self):
        self._lock = threading.Lock()
        self._active_request_ids: set[str] = set()

    def start_request(self, request_id: str) -> None:
        with self._lock:
            self._active_request_ids.add(request_id)

    def finish_request(self, request_id: str) -> bool:
        with self._lock:
            was_active = bool(self._active_request_ids)
            self._active_request_ids.discard(request_id)
            return was_active and not self._active_request_ids

    def is_active(self) -> bool:
        with self._lock:
            return bool(self._active_request_ids)


_VLLM_ACTIVITY = _RolloutPerfVLLMActivity()
_VLLM_METRIC_COLLECTORS_LOCK = threading.Lock()
_VLLM_METRIC_COLLECTORS: list[Any] = []


def _register_vllm_metric_collector(collector: Any) -> None:
    with _VLLM_METRIC_COLLECTORS_LOCK:
        _VLLM_METRIC_COLLECTORS.append(collector)


def _emit_vllm_rollout_finished() -> None:
    with _VLLM_METRIC_COLLECTORS_LOCK:
        collectors = list(_VLLM_METRIC_COLLECTORS)
    for collector in collectors:
        collector.emit_rollout_finished()


class RolloutPerfVLLMSampledMetrics:
    """Coalesce high-frequency vLLM StatLogger records into sampled counters."""

    STATE_COUNTERS = (
        "vllm/scheduler/requests_running",
        "vllm/scheduler/requests_waiting",
        "vllm/kv_cache/usage_ratio",
    )
    BUCKET_SUM_COUNTERS = (
        "vllm/iteration/prefill_requests",
        "vllm/iteration/decode_requests",
        "vllm/iteration/batch_requests_total",
        "vllm/iteration/prefill_tokens_computed",
        "vllm/iteration/decode_tokens",
        "vllm/iteration/batch_tokens_total",
    )

    def __init__(self, context: dict[str, Any], sample_interval_ms: int):
        self.context = dict(context)
        self.sample_interval_ms = _positive_int(sample_interval_ms, 500)
        self.sample_interval_ns = self.sample_interval_ms * 1_000_000
        self._lock = threading.Lock()
        self._latest_state: dict[str, float] = {}
        self._bucket_sums: dict[str, float] = {name: 0.0 for name in self.BUCKET_SUM_COUNTERS}
        self._has_bucket_data = False
        self._next_emit_monotonic_ns: int | None = None
        self._seen_rollout_activity = False
        self._emitted_idle_zero = True
        _register_vllm_metric_collector(self)

    def record_scheduler(
        self,
        *,
        running_requests: Optional[int],
        waiting_requests: Optional[int],
        kv_cache_usage_ratio: Optional[float],
        payload: dict[str, Any],
    ) -> None:
        if not _VLLM_ACTIVITY.is_active():
            self.discard_idle_sample()
            return
        with self._lock:
            self._seen_rollout_activity = True
            self._emitted_idle_zero = False
            self._set_next_emit_if_needed()
            if running_requests is not None:
                self._latest_state["vllm/scheduler/requests_running"] = float(running_requests)
            if waiting_requests is not None:
                self._latest_state["vllm/scheduler/requests_waiting"] = float(waiting_requests)
            if kv_cache_usage_ratio is not None:
                # Keep the peak usage observed in the sampling window. A max is more useful
                # than the last value for spotting short cache pressure spikes.
                key = "vllm/kv_cache/usage_ratio"
                self._latest_state[key] = max(float(kv_cache_usage_ratio), self._latest_state.get(key, 0.0))
            self._emit_due_locked(payload)

    def record_iteration(
        self,
        *,
        prefill_requests: int,
        decode_requests: int,
        batch_requests_total: int,
        prefill_tokens_computed: int,
        decode_tokens: int,
        batch_tokens_total: int,
        payload: dict[str, Any],
    ) -> None:
        if not _VLLM_ACTIVITY.is_active():
            self.discard_idle_sample()
            return
        with self._lock:
            self._seen_rollout_activity = True
            self._emitted_idle_zero = False
            self._set_next_emit_if_needed()
            updates = {
                "vllm/iteration/prefill_requests": prefill_requests,
                "vllm/iteration/decode_requests": decode_requests,
                "vllm/iteration/batch_requests_total": batch_requests_total,
                "vllm/iteration/prefill_tokens_computed": prefill_tokens_computed,
                "vllm/iteration/decode_tokens": decode_tokens,
                "vllm/iteration/batch_tokens_total": batch_tokens_total,
            }
            for name, value in updates.items():
                self._bucket_sums[name] += float(value or 0)
            self._has_bucket_data = True
            self._emit_due_locked(payload)

    def flush(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._emit_locked(payload)

    def discard_idle_sample(self) -> None:
        with self._lock:
            if self._emitted_idle_zero:
                self._reset_locked()

    def emit_rollout_finished(self) -> None:
        with self._lock:
            if self._has_bucket_data:
                self._emit_bucket_sums_locked(self._base_payload({}))
            if self._seen_rollout_activity and not self._emitted_idle_zero:
                payload = self._base_payload({})
                for name in self.STATE_COUNTERS:
                    emit_unsampled_counter(name, 0.0, payload, context=payload)
                self._emitted_idle_zero = True
            self._reset_locked()
            self._seen_rollout_activity = False

    def _set_next_emit_if_needed(self) -> None:
        if self._next_emit_monotonic_ns is None:
            self._next_emit_monotonic_ns = time.monotonic_ns() + self.sample_interval_ns

    def _emit_due_locked(self, payload: dict[str, Any]) -> None:
        if self._next_emit_monotonic_ns is None:
            return
        if time.monotonic_ns() < self._next_emit_monotonic_ns:
            return
        self._emit_locked(payload)
        # If vLLM callbacks were delayed beyond one interval, advance from now so
        # stale data does not create a burst of backfilled samples.
        self._next_emit_monotonic_ns = time.monotonic_ns() + self.sample_interval_ns

    def _base_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        sample_payload = dict(self.context)
        sample_payload.update(payload)
        sample_payload["sample_interval_ms"] = self.sample_interval_ms
        return sample_payload

    def _emit_locked(self, payload: dict[str, Any]) -> None:
        sample_payload = self._base_payload(payload)
        for name in self.STATE_COUNTERS:
            if name in self._latest_state:
                emit_unsampled_counter(name, self._latest_state[name], sample_payload, context=sample_payload)
        if self._has_bucket_data:
            self._emit_bucket_sums_locked(sample_payload)
        # kv_cache usage is max-over-window, so reset it after each emitted window.
        self._latest_state.pop("vllm/kv_cache/usage_ratio", None)

    def _emit_bucket_sums_locked(self, payload: dict[str, Any]) -> None:
        for name in self.BUCKET_SUM_COUNTERS:
            emit_unsampled_counter(name, self._bucket_sums.get(name, 0.0), payload, context=payload)
        self._bucket_sums = {name: 0.0 for name in self.BUCKET_SUM_COUNTERS}
        self._has_bucket_data = False

    def _reset_locked(self) -> None:
        self._latest_state.clear()
        self._bucket_sums = {name: 0.0 for name in self.BUCKET_SUM_COUNTERS}
        self._has_bucket_data = False
        self._next_emit_monotonic_ns = None


class RolloutPerfRequestKVMetricsSampler:
    """Sample active request KV lengths from server-side request progress."""

    LOGICAL_AVG = "vllm/kv_len_logical_avg"
    LOGICAL_P95 = "vllm/kv_len_logical_p95"
    ALLOC_EST_AVG = "vllm/kv_len_alloc_est_avg"
    ALLOC_EST_P95 = "vllm/kv_len_alloc_est_p95"

    def __init__(self, *, context: dict[str, Any], sample_interval_ms: int, block_size: Optional[int] = None):
        self.context = dict(context)
        self.sample_interval_ms = _positive_int(sample_interval_ms, 500)
        self.block_size = _positive_int(block_size, 16)
        self._interval_s = self.sample_interval_ms / 1000.0
        self._lock = threading.Lock()
        self._prompt_lengths: dict[str, int] = {}
        self._logical_lengths: dict[str, int] = {}
        self._last_emit_had_active = False
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="rollout-perf-vllm-kv-sampler", daemon=True)
        self._thread.start()
        payload = dict(self.context)
        payload.update({"sample_interval_ms": self.sample_interval_ms, "block_size": self.block_size})
        emit_unsampled_event("vllm_request_kv_sampler_started", payload, context=payload)

    def set_block_size(self, block_size: Any) -> None:
        value = _positive_int(block_size, self.block_size)
        with self._lock:
            self.block_size = value

    def start_request(self, request_id: str, prompt_token_count: int) -> None:
        _VLLM_ACTIVITY.start_request(request_id)
        prompt_length = max(0, int(prompt_token_count or 0))
        with self._lock:
            self._prompt_lengths[request_id] = prompt_length
            self._logical_lengths[request_id] = prompt_length

    def update_request(self, request_id: str, cumulative_decoded_tokens: int) -> None:
        decoded = max(0, int(cumulative_decoded_tokens or 0))
        with self._lock:
            prompt_length = self._prompt_lengths.get(request_id)
            if prompt_length is None:
                return
            self._logical_lengths[request_id] = prompt_length + decoded

    def finish_request(self, request_id: str) -> None:
        emit_zero = False
        with self._lock:
            self._prompt_lengths.pop(request_id, None)
            self._logical_lengths.pop(request_id, None)
            emit_zero = not self._logical_lengths and self._last_emit_had_active
        if emit_zero:
            self._emit_snapshot(force=True)
        if _VLLM_ACTIVITY.finish_request(request_id):
            _emit_vllm_rollout_finished()

    def close(self) -> None:
        self._closed = True
        self._emit_snapshot(force=True)

    def _run(self) -> None:
        while not self._closed:
            time.sleep(self._interval_s)
            self._emit_snapshot()

    def _emit_snapshot(self, *, force: bool = False) -> None:
        with self._lock:
            logical_lengths = [float(value) for value in self._logical_lengths.values()]
            block_size = self.block_size
            should_emit = force or bool(logical_lengths) or self._last_emit_had_active
            if not should_emit:
                return
            self._last_emit_had_active = bool(logical_lengths)
        allocated_lengths = [float(int(math.ceil(value / block_size) * block_size)) for value in logical_lengths]
        payload = dict(self.context)
        payload.update({"sample_interval_ms": self.sample_interval_ms, "block_size": block_size})
        emit_unsampled_counter(self.LOGICAL_AVG, _mean(logical_lengths), payload, context=payload)
        emit_unsampled_counter(self.LOGICAL_P95, _nearest_rank_percentile(logical_lengths, 95.0), payload, context=payload)
        emit_unsampled_counter(self.ALLOC_EST_AVG, _mean(allocated_lengths), payload, context=payload)
        emit_unsampled_counter(self.ALLOC_EST_P95, _nearest_rank_percentile(allocated_lengths, 95.0), payload, context=payload)


class RolloutPerfVLLMStatLogger(StatLoggerBase):
    """Custom vLLM stat logger that writes engine-level rollout perf counters."""

    def __init__(self, vllm_config: Any, *args: Any, engine_index: int = 0, **kwargs: Any):
        if args and isinstance(args[-1], int):
            engine_index = args[-1]
        engine_index = kwargs.get("engine_index", engine_index)
        self.vllm_config = vllm_config
        self.engine_index = engine_index
        self._capability_reported = False
        self._last_sleep_state: tuple[int, int] | None = None
        self.sample_interval_ms = _positive_int(
            os.getenv("VERL_ROLLOUT_PERF_ENGINE_INTERNAL_SAMPLE_INTERVAL_MS"), 500
        )

        output_dir = os.getenv("VERL_ROLLOUT_PERF_OUTPUT_DIR")
        flush_interval_s = float(os.getenv("VERL_ROLLOUT_PERF_FLUSH_INTERVAL_S", "5.0"))
        init_rollout_perf(
            {
                "enable": True,
                "output_dir": output_dir,
                "format": "jsonl",
                "capture_content": False,
                "flush_interval_s": flush_interval_s,
            },
            role="vllm_engine_core",
        )

        cache_config = _getattr(vllm_config, "cache_config", None)
        scheduler_config = _getattr(vllm_config, "scheduler_config", None)
        self.num_gpu_blocks = _as_int(_config_get(cache_config, "num_gpu_blocks", None))
        self.block_size = _as_int(_config_get(cache_config, "block_size", None))
        self.max_num_batched_tokens = _as_int(_config_get(scheduler_config, "max_num_batched_tokens", None))
        self.max_num_seqs = _as_int(_config_get(scheduler_config, "max_num_seqs", None))
        self.context = {
            "engine_backend": "vllm",
            "engine_index": engine_index,
            "replica_rank": os.getenv("VERL_REPLICA_RANK"),
            "node_rank": os.getenv("VERL_NODE_RANK"),
            "vllm_version": getattr(vllm, "__version__", None) if vllm is not None else None,
        }
        self._sampled_metrics = RolloutPerfVLLMSampledMetrics(self.context, self.sample_interval_ms)

    def log_engine_initialized(self):
        emit_unsampled_event(
            "vllm_engine_internal_logger_started",
            {
                "engine_index": self.engine_index,
                "num_gpu_blocks": self.num_gpu_blocks,
                "block_size": self.block_size,
                "max_num_batched_tokens": self.max_num_batched_tokens,
                "max_num_seqs": self.max_num_seqs,
                "sample_interval_ms": self.sample_interval_ms,
                "stat_logger_interface": "vllm.v1.metrics.loggers.StatLoggerBase",
            },
            context=self.context,
        )

    def record(
        self,
        scheduler_stats: Any,
        iteration_stats: Any,
        mm_cache_stats: Any = None,
        engine_idx: int = 0,
    ):
        payload = dict(self.context)
        payload["engine_idx"] = engine_idx
        if scheduler_stats is not None:
            self._record_scheduler_stats(scheduler_stats, payload)
        if iteration_stats is not None:
            self._record_iteration_stats(iteration_stats, payload)
        if mm_cache_stats is not None:
            self._record_mm_cache_stats(mm_cache_stats, payload)
        if not self._capability_reported:
            self._emit_capability_event(scheduler_stats, iteration_stats, mm_cache_stats, payload)
            self._capability_reported = True

    def log(self):
        pass

    def record_sleep_state(self, is_awake: int, level: int):
        state = (int(is_awake), int(level))
        if state == self._last_sleep_state:
            return
        self._last_sleep_state = state
        payload = dict(self.context)
        payload["sleep_level"] = int(level)
        emit_unsampled_counter("vllm/engine/is_awake", int(is_awake), payload, context=payload)

    def _emit_counter(self, name: str, value: Any, payload: dict[str, Any]):
        number = _as_number(value)
        if number is None:
            return
        emit_unsampled_counter(name, number, payload, context=payload)

    def _record_scheduler_stats(self, stats: Any, payload: dict[str, Any]):
        running = _as_int(_getattr(stats, "num_running_reqs", None))
        waiting = _as_int(_getattr(stats, "num_waiting_reqs", None))
        kv_usage = _as_number(_getattr(stats, "kv_cache_usage", None))
        self._sampled_metrics.record_scheduler(
            running_requests=running,
            waiting_requests=waiting,
            kv_cache_usage_ratio=kv_usage,
            payload=payload,
        )

    def _record_iteration_stats(self, stats: Any, payload: dict[str, Any]):
        prompt_stats = _getattr(stats, "prompt_token_stats", None)
        prefill_tokens_total = _as_int(_getattr(prompt_stats, "total", None))
        prefill_tokens_computed = _as_int(_getattr(prompt_stats, "computed", None))
        if prefill_tokens_total is None:
            prefill_tokens_total = _as_int(_getattr(stats, "num_prompt_tokens", None))

        prefill_requests = _safe_len(_getattr(stats, "time_to_first_tokens_iter", None)) or 0
        decode_requests = _safe_len(_getattr(stats, "inter_token_latencies_iter", None)) or 0
        generation_tokens = _as_int(_getattr(stats, "num_generation_tokens", None)) or 0
        decode_tokens = max(generation_tokens - prefill_requests, 0)

        computed_prefill = prefill_tokens_computed if prefill_tokens_computed is not None else (prefill_tokens_total or 0)
        batch_tokens_total = computed_prefill + decode_tokens
        batch_requests_total = prefill_requests + decode_requests

        self._sampled_metrics.record_iteration(
            prefill_requests=prefill_requests,
            decode_requests=decode_requests,
            batch_requests_total=batch_requests_total,
            prefill_tokens_computed=computed_prefill,
            decode_tokens=decode_tokens,
            batch_tokens_total=batch_tokens_total,
            payload=payload,
        )

    def _record_mm_cache_stats(self, stats: Any, payload: dict[str, Any]):
        self._emit_counter("vllm/mm_cache/hit_rate", _hit_rate(stats), payload)

    def _emit_capability_event(
        self, scheduler_stats: Any, iteration_stats: Any, mm_cache_stats: Any, payload: dict[str, Any]
    ):
        prompt_stats = _getattr(iteration_stats, "prompt_token_stats", None)
        available = {
            "scheduler_stats": scheduler_stats is not None,
            "iteration_stats": iteration_stats is not None,
            "mm_cache_stats": mm_cache_stats is not None,
            "requests_running": _getattr(scheduler_stats, "num_running_reqs", None) is not None,
            "requests_waiting": _getattr(scheduler_stats, "num_waiting_reqs", None) is not None,
            "kv_cache_usage_ratio": _getattr(scheduler_stats, "kv_cache_usage", None) is not None,
            "kv_blocks_total": self.num_gpu_blocks is not None,
            "prefill_token_stats": prompt_stats is not None,
            "generation_tokens": _getattr(iteration_stats, "num_generation_tokens", None) is not None,
            "prefill_requests": _getattr(iteration_stats, "time_to_first_tokens_iter", None) is not None,
            "decode_requests": _getattr(iteration_stats, "inter_token_latencies_iter", None) is not None,
            "preempted_requests": _getattr(iteration_stats, "num_preempted_reqs", None) is not None,
        }
        missing_or_deferred = []
        emit_unsampled_event(
            "vllm_engine_internal_metrics_capability",
            {
                "available": available,
                "missing_or_deferred": missing_or_deferred,
                "notes": (
                    "Scheduler and iteration metrics come from vLLM StatLogger and are sampled in-process. "
                    "KV length avg/p95 metrics are sampled from server-side request progress."
                ),
            },
            context=payload,
        )
