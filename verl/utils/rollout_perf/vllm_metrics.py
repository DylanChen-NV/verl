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

import os
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

    def log_engine_initialized(self):
        emit_unsampled_event(
            "vllm_engine_internal_logger_started",
            {
                "engine_index": self.engine_index,
                "num_gpu_blocks": self.num_gpu_blocks,
                "block_size": self.block_size,
                "max_num_batched_tokens": self.max_num_batched_tokens,
                "max_num_seqs": self.max_num_seqs,
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
        skipped_waiting = _as_int(_getattr(stats, "num_skipped_waiting_reqs", None))
        kv_usage = _as_number(_getattr(stats, "kv_cache_usage", None))

        self._emit_counter("vllm/scheduler/running_requests", running, payload)
        self._emit_counter("vllm/scheduler/waiting_requests", waiting, payload)
        self._emit_counter("vllm/scheduler/skipped_waiting_requests", skipped_waiting, payload)
        if waiting is not None or skipped_waiting is not None:
            self._emit_counter("vllm/scheduler/waiting_requests_total", (waiting or 0) + (skipped_waiting or 0), payload)
        self._emit_counter("vllm/kv_cache/usage_ratio", kv_usage, payload)

        if kv_usage is not None and self.num_gpu_blocks is not None:
            used_blocks = int(round(kv_usage * self.num_gpu_blocks))
            self._emit_counter("vllm/kv_cache/blocks_total", self.num_gpu_blocks, payload)
            self._emit_counter("vllm/kv_cache/blocks_used", used_blocks, payload)
            self._emit_counter("vllm/kv_cache/blocks_free", max(self.num_gpu_blocks - used_blocks, 0), payload)

        prefix_rate = _hit_rate(_getattr(stats, "prefix_cache_stats", None))
        self._emit_counter("vllm/prefix_cache/hit_rate", prefix_rate, payload)

        spec_stats = _getattr(stats, "spec_decoding_stats", None)
        if spec_stats is not None:
            self._emit_counter("vllm/spec_decode/accepted_tokens", _getattr(spec_stats, "num_accepted_tokens", None), payload)
            self._emit_counter("vllm/spec_decode/draft_tokens", _getattr(spec_stats, "num_draft_tokens", None), payload)

    def _record_iteration_stats(self, stats: Any, payload: dict[str, Any]):
        prompt_stats = _getattr(stats, "prompt_token_stats", None)
        prefill_tokens_total = _as_int(_getattr(prompt_stats, "total", None))
        prefill_tokens_computed = _as_int(_getattr(prompt_stats, "computed", None))
        prefill_tokens_cached = _as_int(_getattr(prompt_stats, "cached_tokens", None))
        if prefill_tokens_total is None:
            prefill_tokens_total = _as_int(_getattr(stats, "num_prompt_tokens", None))

        prefill_requests = _safe_len(_getattr(stats, "time_to_first_tokens_iter", None)) or 0
        decode_requests = _safe_len(_getattr(stats, "inter_token_latencies_iter", None)) or 0
        generation_tokens = _as_int(_getattr(stats, "num_generation_tokens", None)) or 0
        decode_tokens = max(generation_tokens - prefill_requests, 0)
        preempted_requests = _as_int(_getattr(stats, "num_preempted_reqs", None))

        computed_prefill = prefill_tokens_computed if prefill_tokens_computed is not None else (prefill_tokens_total or 0)
        batch_tokens_total = computed_prefill + decode_tokens
        batch_requests_total = prefill_requests + decode_requests

        self._emit_counter("vllm/iteration/prefill_requests", prefill_requests, payload)
        self._emit_counter("vllm/iteration/decode_requests", decode_requests, payload)
        self._emit_counter("vllm/iteration/batch_requests_total", batch_requests_total, payload)
        self._emit_counter("vllm/iteration/prefill_tokens_total", prefill_tokens_total, payload)
        self._emit_counter("vllm/iteration/prefill_tokens_computed", prefill_tokens_computed, payload)
        self._emit_counter("vllm/iteration/prefill_tokens_cached", prefill_tokens_cached, payload)
        self._emit_counter("vllm/iteration/decode_tokens", decode_tokens, payload)
        self._emit_counter("vllm/iteration/batch_tokens_total", batch_tokens_total, payload)
        self._emit_counter("vllm/iteration/preempted_requests", preempted_requests, payload)

        self._emit_counter("vllm/iteration/prefill_tokens_local_cache_hit", _getattr(prompt_stats, "local_cache_hit", None), payload)
        self._emit_counter("vllm/iteration/prefill_tokens_external_kv", _getattr(prompt_stats, "external_kv_transfer", None), payload)

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
            "running_requests": _getattr(scheduler_stats, "num_running_reqs", None) is not None,
            "waiting_requests": _getattr(scheduler_stats, "num_waiting_reqs", None) is not None,
            "kv_cache_usage_ratio": _getattr(scheduler_stats, "kv_cache_usage", None) is not None,
            "kv_blocks_total": self.num_gpu_blocks is not None,
            "prefill_token_stats": prompt_stats is not None,
            "generation_tokens": _getattr(iteration_stats, "num_generation_tokens", None) is not None,
            "prefill_requests": _getattr(iteration_stats, "time_to_first_tokens_iter", None) is not None,
            "decode_requests": _getattr(iteration_stats, "inter_token_latencies_iter", None) is not None,
            "preempted_requests": _getattr(iteration_stats, "num_preempted_reqs", None) is not None,
        }
        missing_or_deferred = [
            "kv_cache_len_avg",
            "kv_cache_len_p50",
            "kv_cache_len_p95",
            "kv_cache_len_max",
            "per_request_active_kv_lengths",
        ]
        emit_unsampled_event(
            "vllm_engine_internal_metrics_capability",
            {
                "available": available,
                "missing_or_deferred": missing_or_deferred,
                "notes": (
                    "Metrics come from vLLM StatLogger SchedulerStats/IterationStats. "
                    "KV length distribution is deferred because it is not exposed by this interface."
                ),
            },
            context=payload,
        )
