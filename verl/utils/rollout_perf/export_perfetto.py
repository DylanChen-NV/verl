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
import argparse
import json
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


AGENT_COUNTER_SPANS = {
    "trajectory",
    "llm_turn",
    "llm_client_request",
    "tool_turn",
    "tool_call",
}

VLLM_SERVER_COUNTER_SPANS = {"vllm_engine_request"}

VLLM_STATE_COUNTERS = {
    "vllm/scheduler/requests_running": ("vllm/requests_running", 1.0),
    "vllm/scheduler/requests_waiting": ("vllm/requests_waiting", 1.0),
    # Backward compatibility for traces collected before the requests_* rename.
    "vllm/scheduler/running_requests": ("vllm/requests_running", 1.0),
    "vllm/scheduler/waiting_requests": ("vllm/requests_waiting", 1.0),
    "vllm/kv_cache/usage_ratio": ("vllm/kv_cache_usage_%", 100.0),
    "vllm/kv_len_per_request_logical_avg": ("vllm/kv_len_per_request_logical_avg", 1.0),
    "vllm/kv_len_per_request_logical_p95": ("vllm/kv_len_per_request_logical_p95", 1.0),
    "vllm/kv_len_per_request_alloc_est_avg": ("vllm/kv_len_per_request_alloc_est_avg", 1.0),
    "vllm/kv_len_per_request_alloc_est_p95": ("vllm/kv_len_per_request_alloc_est_p95", 1.0),
    # Backward compatibility for traces collected before the per-request rename.
    "vllm/kv_len_logical_avg": ("vllm/kv_len_per_request_logical_avg", 1.0),
    "vllm/kv_len_logical_p95": ("vllm/kv_len_per_request_logical_p95", 1.0),
    "vllm/kv_len_alloc_est_avg": ("vllm/kv_len_per_request_alloc_est_avg", 1.0),
    "vllm/kv_len_alloc_est_p95": ("vllm/kv_len_per_request_alloc_est_p95", 1.0),
}

GLOBAL_VLLM_ENGINE_SUM_COUNTERS = {
    "vllm/requests_running",
    "vllm/requests_waiting",
    "vllm/requests_batch",
    "vllm/requests_prefill",
    "vllm/requests_decode",
    "vllm/tokens_batch",
    "vllm/tokens_prefill",
    "vllm/tokens_decode",
}

GLOBAL_VLLM_ENGINE_DISTRIBUTION_COUNTERS = {
    "vllm/kv_cache_usage_%": ("vllm/kv_cache_usage_%_mean", "vllm/kv_cache_usage_%_max"),
    "vllm/kv_len_per_request_logical_avg": (
        "vllm/kv_len_per_request_logical_avg_mean",
        "vllm/kv_len_per_request_logical_avg_max",
    ),
    "vllm/kv_len_per_request_logical_p95": (
        "vllm/kv_len_per_request_logical_p95_mean",
        "vllm/kv_len_per_request_logical_p95_max",
    ),
    "vllm/kv_len_per_request_alloc_est_avg": (
        "vllm/kv_len_per_request_alloc_est_avg_mean",
        "vllm/kv_len_per_request_alloc_est_avg_max",
    ),
    "vllm/kv_len_per_request_alloc_est_p95": (
        "vllm/kv_len_per_request_alloc_est_p95_mean",
        "vllm/kv_len_per_request_alloc_est_p95_max",
    ),
}

VLLM_BUCKET_SUM_COUNTERS = {
    "vllm/iteration/batch_requests_total": "vllm/requests_batch",
    "vllm/iteration/prefill_requests": "vllm/requests_prefill",
    "vllm/iteration/decode_requests": "vllm/requests_decode",
    "vllm/iteration/batch_tokens_total": "vllm/tokens_batch",
    "vllm/iteration/prefill_tokens_computed": "vllm/tokens_prefill",
    "vllm/iteration/decode_tokens": "vllm/tokens_decode",
}

VLLM_FOCUSED_COUNTERS = set(VLLM_STATE_COUNTERS) | set(VLLM_BUCKET_SUM_COUNTERS)


def _iter_records(input_path: Path) -> Iterable[dict[str, Any]]:
    paths = [input_path]
    if input_path.is_dir():
        paths = sorted(input_path.glob("rollout_perf_*.jsonl"))
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def _thread_name(record: dict[str, Any]) -> str:
    context = record.get("context") or {}
    payload = record.get("payload") or {}
    for key in ("engine_request_id", "logical_request_id", "trajectory_id", "server_id", "worker_id"):
        value = payload.get(key) or context.get(key)
        if value is not None:
            return f"{key}:{value}"
    return record.get("role", "unknown")


def _process_name(record: dict[str, Any]) -> str:
    return f"{record.get('role', 'unknown')}@{record.get('hostname', 'host')}:{record.get('pid', 0)}"


def _base_args(record: dict[str, Any]) -> dict[str, Any]:
    args = dict(record.get("payload") or {})
    context = record.get("context") or {}
    for key, value in context.items():
        args.setdefault(key, value)
    args.setdefault("status", record.get("status"))
    args.setdefault("priority", record.get("priority"))
    return args


def records_to_perfetto(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    events = []
    process_ids: dict[str, int] = {}
    thread_ids: dict[tuple[str, str], int] = {}

    def get_pid(record: dict[str, Any]) -> int:
        process = _process_name(record)
        if process not in process_ids:
            process_ids[process] = len(process_ids) + 1
            events.append({"name": "process_name", "ph": "M", "pid": process_ids[process], "args": {"name": process}})
        return process_ids[process]

    def get_tid(record: dict[str, Any], pid: int) -> int:
        thread = _thread_name(record)
        key = (str(pid), thread)
        if key not in thread_ids:
            thread_ids[key] = len(thread_ids) + 1
            events.append({"name": "thread_name", "ph": "M", "pid": pid, "tid": thread_ids[key], "args": {"name": thread}})
        return thread_ids[key]

    for record in records:
        record_type = record.get("record_type")
        if record_type not in {"span", "event", "counter"}:
            continue
        pid = get_pid(record)
        tid = get_tid(record, pid)
        args = _base_args(record)
        if record_type == "span":
            events.append(
                {
                    "name": record.get("name", "span"),
                    "cat": "rollout_perf",
                    "ph": "X",
                    "ts": record.get("start_unix_ns", record.get("time_unix_ns", 0)) / 1000,
                    "dur": record.get("duration_ns", 0) / 1000,
                    "pid": pid,
                    "tid": tid,
                    "args": args,
                }
            )
        elif record_type == "event":
            events.append(
                {
                    "name": record.get("name", "event"),
                    "cat": "rollout_perf",
                    "ph": "i",
                    "s": "t",
                    "ts": record.get("time_unix_ns", 0) / 1000,
                    "pid": pid,
                    "tid": tid,
                    "args": args,
                }
            )
        elif record_type == "counter":
            events.append(
                {
                    "name": record.get("name", "counter"),
                    "cat": "rollout_perf",
                    "ph": "C",
                    "ts": record.get("time_unix_ns", 0) / 1000,
                    "pid": pid,
                    "tid": tid,
                    "args": args,
                }
            )
    return {"traceEvents": events}


def _span_bounds(record: dict[str, Any]) -> tuple[int, int] | None:
    start = record.get("start_unix_ns")
    end = record.get("end_unix_ns")
    if start is None:
        return None
    if end is None:
        duration = record.get("duration_ns")
        if duration is None:
            return None
        end = start + duration
    if end < start:
        return None
    return int(start), int(end)


def _source_key(record: dict[str, Any]) -> tuple[str, int]:
    return str(record.get("hostname", "host")), int(record.get("pid", 0) or 0)


def _stable_id_map(keys: Iterable[tuple[str, int]], prefix: str) -> dict[tuple[str, int], str]:
    return {key: f"{prefix}{index}" for index, key in enumerate(sorted(set(keys)))}


def _engine_index(record: dict[str, Any]) -> str:
    payload = record.get("payload") or {}
    context = record.get("context") or {}
    for key in ("engine_idx", "engine_index"):
        value = payload.get(key, context.get(key))
        if value is not None:
            return str(value)
    return "0"


def _collect_active_counter_ids(
    records: Iterable[dict[str, Any]],
) -> tuple[dict[tuple[str, int], str], dict[tuple[str, int], str]]:
    agent_keys = []
    server_keys = []
    for record in records:
        record_type = record.get("record_type")
        name = record.get("name")
        role = record.get("role")
        if record_type == "span" and role == "agent_loop_worker" and name in AGENT_COUNTER_SPANS:
            agent_keys.append(_source_key(record))
        if record_type == "span" and name in VLLM_SERVER_COUNTER_SPANS:
            server_keys.append(_source_key(record))
        if record_type == "counter" and name in VLLM_FOCUSED_COUNTERS:
            server_keys.append(_source_key(record))
    return _stable_id_map(agent_keys, "a"), _stable_id_map(server_keys, "s")


def _add_counter_metadata(events: list[dict[str, Any]], process_ids: dict[str, int], process: str) -> int:
    if process not in process_ids:
        process_ids[process] = len(process_ids) + 1
        pid = process_ids[process]
        events.append({"name": "process_name", "ph": "M", "pid": pid, "args": {"name": process}})
        events.append({"name": "thread_name", "ph": "M", "pid": pid, "tid": 1, "args": {"name": "counters"}})
    return process_ids[process]


def _process_span_groups(
    records: Iterable[dict[str, Any]],
    agent_ids: dict[tuple[str, int], str],
    server_ids: dict[tuple[str, int], str],
) -> dict[tuple[str, str, str], list[tuple[int, int]]]:
    groups: dict[tuple[str, str, str], list[tuple[int, int]]] = {}
    for record in records:
        if record.get("record_type") != "span":
            continue
        span_name = record.get("name")
        bounds = _span_bounds(record)
        if bounds is None:
            continue
        key = _source_key(record)
        if record.get("role") == "agent_loop_worker" and span_name in AGENT_COUNTER_SPANS and key in agent_ids:
            group_id = agent_ids[key]
            groups.setdefault((f"agent_loop_worker/{group_id}", span_name, group_id), []).append(bounds)
        elif span_name in VLLM_SERVER_COUNTER_SPANS and key in server_ids:
            group_id = server_ids[key]
            groups.setdefault((f"vllm_server/{group_id}", span_name, group_id), []).append(bounds)
    return groups


def _global_span_groups(records: Iterable[dict[str, Any]]) -> dict[tuple[str, str], list[tuple[int, int]]]:
    groups: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for record in records:
        if record.get("record_type") != "span":
            continue
        span_name = record.get("name")
        bounds = _span_bounds(record)
        if bounds is None:
            continue
        if record.get("role") == "agent_loop_worker" and span_name in AGENT_COUNTER_SPANS:
            groups.setdefault(("global/agent_loop_worker", span_name), []).append(bounds)
        elif span_name in VLLM_SERVER_COUNTER_SPANS:
            groups.setdefault(("global/vllm_server", span_name), []).append(bounds)
    return groups


def _add_global_span_counter_events(
    events: list[dict[str, Any]],
    process_ids: dict[str, int],
    records: Iterable[dict[str, Any]],
    sample_interval_ns: int,
) -> None:
    for (process, span_name), spans in sorted(_global_span_groups(records).items()):
        pid = _add_counter_metadata(events, process_ids, process)
        for sample_ts, active in _iter_counter_samples(spans, sample_interval_ns):
            events.append(
                {
                    "name": f"active/{span_name}",
                    "cat": "rollout_perf_active_global",
                    "ph": "C",
                    "ts": sample_ts / 1000,
                    "pid": pid,
                    "tid": 1,
                    "args": _active_counter_args(active),
                }
            )


def _iter_counter_samples(
    spans: list[tuple[int, int]],
    sample_interval_ns: int,
) -> Iterable[tuple[int, int]]:
    if not spans:
        return
    events = []
    min_start = min(start for start, _ in spans)
    max_end = max(end for _, end in spans)
    for start, end in spans:
        events.append((start, 1))
        events.append((end, -1))
    events.sort(key=lambda item: (item[0], item[1]))

    event_index = 0
    active = 0

    def process_events_until(timestamp: int) -> int:
        nonlocal event_index, active
        while event_index < len(events) and events[event_index][0] <= timestamp:
            active += events[event_index][1]
            event_index += 1
        return active

    last_sample_ts = min_start
    yield last_sample_ts, process_events_until(last_sample_ts)

    sample_ts = ((min_start // sample_interval_ns) + 1) * sample_interval_ns
    while sample_ts < max_end:
        yield sample_ts, process_events_until(sample_ts)
        last_sample_ts = sample_ts
        sample_ts += sample_interval_ns

    if max_end != last_sample_ts:
        yield max_end, process_events_until(max_end)


def _active_counter_args(active: int) -> dict[str, Any]:
    return {"value": active}


def _counter_args(value: float) -> dict[str, Any]:
    return {"value": value}


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _trace_time_bounds(records: Iterable[dict[str, Any]]) -> tuple[int, int] | None:
    min_ts = None
    max_ts = None
    for record in records:
        bounds = _span_bounds(record) if record.get("record_type") == "span" else None
        if bounds is not None:
            candidates = bounds
        else:
            timestamp = record.get("time_unix_ns")
            if timestamp is None:
                continue
            candidates = (int(timestamp), int(timestamp))
        for timestamp in candidates:
            min_ts = timestamp if min_ts is None else min(min_ts, timestamp)
            max_ts = timestamp if max_ts is None else max(max_ts, timestamp)
    if min_ts is None or max_ts is None:
        return None
    return min_ts, max_ts


def _iter_sample_timestamps(start_ns: int, end_ns: int, sample_interval_ns: int) -> Iterable[int]:
    sample_ts = start_ns
    while sample_ts < end_ns:
        yield sample_ts
        sample_ts += sample_interval_ns
    yield end_ns


def _merge_intervals(spans: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if end < start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _iter_sample_timestamps_for_spans(
    spans: list[tuple[int, int]],
    sample_interval_ns: int,
) -> Iterable[int]:
    for start, end in _merge_intervals(spans):
        last_sample_ts = start
        yield start
        sample_ts = ((start // sample_interval_ns) + 1) * sample_interval_ns
        while sample_ts < end:
            yield sample_ts
            last_sample_ts = sample_ts
            sample_ts += sample_interval_ns
        if end != last_sample_ts:
            yield end


def _timestamp_in_spans(timestamp: int, spans: list[tuple[int, int]]) -> bool:
    return any(start <= timestamp <= end for start, end in spans)


def _server_id_from_engine_id(engine_id: str) -> str:
    return engine_id.split("e", 1)[0]


def _collect_server_active_spans(
    records: Iterable[dict[str, Any]],
    server_ids: dict[tuple[str, int], str],
) -> dict[str, list[tuple[int, int]]]:
    spans_by_server: dict[str, list[tuple[int, int]]] = {}
    for record in records:
        if record.get("record_type") != "span" or record.get("name") not in VLLM_SERVER_COUNTER_SPANS:
            continue
        key = _source_key(record)
        server_id = server_ids.get(key)
        bounds = _span_bounds(record)
        if server_id is None or bounds is None:
            continue
        spans_by_server.setdefault(server_id, []).append(bounds)
    return {server_id: _merge_intervals(spans) for server_id, spans in spans_by_server.items()}


def _counter_sample_timestamps_for_engine(
    engine_id: str,
    server_active_spans: dict[str, list[tuple[int, int]]],
    trace_bounds: tuple[int, int] | None,
    sample_interval_ns: int,
) -> list[int]:
    server_id = _server_id_from_engine_id(engine_id)
    spans = server_active_spans.get(server_id)
    if spans:
        return list(_iter_sample_timestamps_for_spans(spans, sample_interval_ns))
    if trace_bounds is None:
        return []
    start_ns, end_ns = trace_bounds
    return list(_iter_sample_timestamps(start_ns, end_ns, sample_interval_ns))


def _engine_id(record: dict[str, Any], server_ids: dict[tuple[str, int], str]) -> str | None:
    source_key = _source_key(record)
    if source_key not in server_ids:
        return None
    return f"{server_ids[source_key]}e{_engine_index(record)}"


def _numeric_payload_value(record: dict[str, Any]) -> float | None:
    value = (record.get("payload") or {}).get("value")
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _collect_vllm_state_series(
    records: Iterable[dict[str, Any]],
    server_ids: dict[tuple[str, int], str],
    server_active_spans: dict[str, list[tuple[int, int]]],
) -> dict[tuple[str, str], list[tuple[int, float]]]:
    series: dict[tuple[str, str], list[tuple[int, float]]] = {}
    for record in records:
        if record.get("record_type") != "counter":
            continue
        mapping = VLLM_STATE_COUNTERS.get(record.get("name"))
        if mapping is None:
            continue
        engine_id = _engine_id(record, server_ids)
        value = _numeric_payload_value(record)
        if engine_id is None or value is None:
            continue
        timestamp = int(record.get("time_unix_ns", 0))
        server_spans = server_active_spans.get(_server_id_from_engine_id(engine_id))
        if server_spans and not _timestamp_in_spans(timestamp, server_spans):
            continue
        output_name, scale = mapping
        series.setdefault((engine_id, output_name), []).append((timestamp, value * scale))
    for samples in series.values():
        samples.sort()
    return series


def _collect_vllm_bucket_series(
    records: Iterable[dict[str, Any]],
    server_ids: dict[tuple[str, int], str],
    server_active_spans: dict[str, list[tuple[int, int]]],
) -> dict[str, list[tuple[int, float]]]:
    raw_series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for record in records:
        if record.get("record_type") != "counter":
            continue
        output_name = VLLM_BUCKET_SUM_COUNTERS.get(record.get("name"))
        if output_name is None:
            continue
        engine_id = _engine_id(record, server_ids)
        value = _numeric_payload_value(record)
        if engine_id is None or value is None:
            continue
        timestamp = int(record.get("time_unix_ns", 0))
        server_spans = server_active_spans.get(_server_id_from_engine_id(engine_id))
        if server_spans and not _timestamp_in_spans(timestamp, server_spans):
            continue
        raw_series[output_name].append((timestamp, value))
    for samples in raw_series.values():
        samples.sort()
    return dict(raw_series)


def _global_vllm_sample_timestamps(
    server_active_spans: dict[str, list[tuple[int, int]]],
    sample_interval_ns: int,
) -> list[int]:
    spans: list[tuple[int, int]] = []
    for server_spans in server_active_spans.values():
        spans.extend(server_spans)
    return list(_iter_sample_timestamps_for_spans(spans, sample_interval_ns))


def _add_vllm_state_counter_events(
    events: list[dict[str, Any]],
    process_ids: dict[str, int],
    records: Iterable[dict[str, Any]],
    server_ids: dict[tuple[str, int], str],
    server_active_spans: dict[str, list[tuple[int, int]]],
    trace_bounds: tuple[int, int] | None,
    sample_interval_ns: int,
) -> None:
    series = _collect_vllm_state_series(records, server_ids, server_active_spans)
    for (engine_id, output_name), samples in sorted(series.items()):
        sample_timestamps = _counter_sample_timestamps_for_engine(
            engine_id, server_active_spans, trace_bounds, sample_interval_ns
        )
        if not sample_timestamps:
            continue
        sample_index = 0
        current_value = 0.0
        pid = _add_counter_metadata(events, process_ids, f"vllm_engine/{engine_id}")
        for sample_ts in sample_timestamps:
            while sample_index < len(samples) and samples[sample_index][0] <= sample_ts:
                current_value = samples[sample_index][1]
                sample_index += 1
            events.append(
                {
                    "name": output_name,
                    "cat": "rollout_perf_counter",
                    "ph": "C",
                    "ts": sample_ts / 1000,
                    "pid": pid,
                    "tid": 1,
                    "args": _counter_args(current_value),
                }
            )


def _add_vllm_bucket_sum_counter_events(
    events: list[dict[str, Any]],
    process_ids: dict[str, int],
    records: Iterable[dict[str, Any]],
    server_ids: dict[tuple[str, int], str],
    server_active_spans: dict[str, list[tuple[int, int]]],
    trace_bounds: tuple[int, int] | None,
    sample_interval_ns: int,
) -> None:
    raw_series: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for record in records:
        if record.get("record_type") != "counter":
            continue
        output_name = VLLM_BUCKET_SUM_COUNTERS.get(record.get("name"))
        if output_name is None:
            continue
        engine_id = _engine_id(record, server_ids)
        value = _numeric_payload_value(record)
        if engine_id is None or value is None:
            continue
        timestamp = int(record.get("time_unix_ns", 0))
        server_spans = server_active_spans.get(_server_id_from_engine_id(engine_id))
        if server_spans and not _timestamp_in_spans(timestamp, server_spans):
            continue
        raw_series[(engine_id, output_name)].append((timestamp, value))

    for (engine_id, output_name), samples in sorted(raw_series.items()):
        sample_timestamps = _counter_sample_timestamps_for_engine(
            engine_id, server_active_spans, trace_bounds, sample_interval_ns
        )
        if not sample_timestamps:
            continue
        buckets = [0.0] * len(sample_timestamps)
        for timestamp, value in sorted(samples):
            bucket_index = bisect_right(sample_timestamps, timestamp) - 1
            bucket_index = max(0, min(bucket_index, len(buckets) - 1))
            buckets[bucket_index] += value
        pid = _add_counter_metadata(events, process_ids, f"vllm_engine/{engine_id}")
        for sample_ts, value in zip(sample_timestamps, buckets):
            events.append(
                {
                    "name": output_name,
                    "cat": "rollout_perf_counter",
                    "ph": "C",
                    "ts": sample_ts / 1000,
                    "pid": pid,
                    "tid": 1,
                    "args": _counter_args(value),
                }
            )


def _add_global_vllm_engine_state_counter_events(
    events: list[dict[str, Any]],
    process_ids: dict[str, int],
    records: Iterable[dict[str, Any]],
    server_ids: dict[tuple[str, int], str],
    server_active_spans: dict[str, list[tuple[int, int]]],
    sample_interval_ns: int,
) -> None:
    sample_timestamps = _global_vllm_sample_timestamps(server_active_spans, sample_interval_ns)
    if not sample_timestamps:
        return
    series = _collect_vllm_state_series(records, server_ids, server_active_spans)
    if not series:
        return
    sample_indices = {key: 0 for key in series}
    current_values = {key: 0.0 for key in series}
    pid = _add_counter_metadata(events, process_ids, "global/vllm_engine")
    for sample_ts in sample_timestamps:
        values_by_name: dict[str, list[float]] = defaultdict(list)
        for key, samples in sorted(series.items()):
            engine_id, output_name = key
            server_spans = server_active_spans.get(_server_id_from_engine_id(engine_id), [])
            if not _timestamp_in_spans(sample_ts, server_spans):
                current_values[key] = 0.0
                continue
            sample_index = sample_indices[key]
            while sample_index < len(samples) and samples[sample_index][0] <= sample_ts:
                current_values[key] = samples[sample_index][1]
                sample_index += 1
            sample_indices[key] = sample_index
            values_by_name[output_name].append(current_values[key])
        for output_name, values in sorted(values_by_name.items()):
            if output_name in GLOBAL_VLLM_ENGINE_SUM_COUNTERS:
                events.append(
                    {
                        "name": output_name,
                        "cat": "rollout_perf_counter_global",
                        "ph": "C",
                        "ts": sample_ts / 1000,
                        "pid": pid,
                        "tid": 1,
                        "args": _counter_args(float(sum(values))),
                    }
                )
            elif output_name in GLOBAL_VLLM_ENGINE_DISTRIBUTION_COUNTERS:
                mean_name, max_name = GLOBAL_VLLM_ENGINE_DISTRIBUTION_COUNTERS[output_name]
                for name, value in ((mean_name, _mean(values)), (max_name, max(values))):
                    events.append(
                        {
                            "name": name,
                            "cat": "rollout_perf_counter_global",
                            "ph": "C",
                            "ts": sample_ts / 1000,
                            "pid": pid,
                            "tid": 1,
                            "args": _counter_args(float(value)),
                        }
                    )


def _add_global_vllm_engine_bucket_counter_events(
    events: list[dict[str, Any]],
    process_ids: dict[str, int],
    records: Iterable[dict[str, Any]],
    server_ids: dict[tuple[str, int], str],
    server_active_spans: dict[str, list[tuple[int, int]]],
    sample_interval_ns: int,
) -> None:
    sample_timestamps = _global_vllm_sample_timestamps(server_active_spans, sample_interval_ns)
    if not sample_timestamps:
        return
    raw_series = _collect_vllm_bucket_series(records, server_ids, server_active_spans)
    if not raw_series:
        return
    pid = _add_counter_metadata(events, process_ids, "global/vllm_engine")
    for output_name, samples in sorted(raw_series.items()):
        buckets = [0.0] * len(sample_timestamps)
        for timestamp, value in samples:
            bucket_index = bisect_right(sample_timestamps, timestamp) - 1
            bucket_index = max(0, min(bucket_index, len(buckets) - 1))
            buckets[bucket_index] += value
        for sample_ts, value in zip(sample_timestamps, buckets):
            events.append(
                {
                    "name": output_name,
                    "cat": "rollout_perf_counter_global",
                    "ph": "C",
                    "ts": sample_ts / 1000,
                    "pid": pid,
                    "tid": 1,
                    "args": _counter_args(value),
                }
            )


def _add_global_vllm_engine_counter_events(
    events: list[dict[str, Any]],
    process_ids: dict[str, int],
    records: Iterable[dict[str, Any]],
    server_ids: dict[tuple[str, int], str],
    server_active_spans: dict[str, list[tuple[int, int]]],
    sample_interval_ns: int,
) -> None:
    _add_global_vllm_engine_state_counter_events(
        events, process_ids, records, server_ids, server_active_spans, sample_interval_ns
    )
    _add_global_vllm_engine_bucket_counter_events(
        events, process_ids, records, server_ids, server_active_spans, sample_interval_ns
    )


def _add_selected_vllm_counter_events(
    events: list[dict[str, Any]],
    process_ids: dict[str, int],
    records: Iterable[dict[str, Any]],
    server_ids: dict[tuple[str, int], str],
    trace_bounds: tuple[int, int] | None,
    sample_interval_ns: int,
) -> None:
    server_active_spans = _collect_server_active_spans(records, server_ids)
    _add_global_vllm_engine_counter_events(
        events, process_ids, records, server_ids, server_active_spans, sample_interval_ns
    )
    _add_vllm_state_counter_events(
        events, process_ids, records, server_ids, server_active_spans, trace_bounds, sample_interval_ns
    )
    _add_vllm_bucket_sum_counter_events(
        events, process_ids, records, server_ids, server_active_spans, trace_bounds, sample_interval_ns
    )


def records_to_active_counter_perfetto(
    records: Iterable[dict[str, Any]],
    *,
    sample_interval_ms: float,
    counter_scope: str,
) -> dict[str, Any]:
    del counter_scope  # Kept for CLI compatibility; active_counters now uses structured groups.

    if sample_interval_ms <= 0:
        raise ValueError("--sample-interval-ms must be positive")
    sample_interval_ns = int(sample_interval_ms * 1_000_000)
    if sample_interval_ns <= 0:
        raise ValueError("--sample-interval-ms is too small")

    record_list = list(records)
    agent_ids, server_ids = _collect_active_counter_ids(record_list)
    groups = _process_span_groups(record_list, agent_ids, server_ids)
    trace_bounds = _trace_time_bounds(record_list)
    events: list[dict[str, Any]] = []
    process_ids: dict[str, int] = {}

    _add_global_span_counter_events(events, process_ids, record_list, sample_interval_ns)
    for (process, span_name, group_id), spans in sorted(groups.items()):
        pid = _add_counter_metadata(events, process_ids, process)
        for sample_ts, active in _iter_counter_samples(spans, sample_interval_ns):
            events.append(
                {
                    "name": f"active/{span_name}",
                    "cat": "rollout_perf_active",
                    "ph": "C",
                    "ts": sample_ts / 1000,
                    "pid": pid,
                    "tid": 1,
                    "args": _active_counter_args(active),
                }
            )
    _add_selected_vllm_counter_events(events, process_ids, record_list, server_ids, trace_bounds, sample_interval_ns)
    return {"traceEvents": events}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export rollout perf JSONL records to Perfetto JSON.")
    parser.add_argument("--input", required=True, help="Input rollout perf JSONL file or directory.")
    parser.add_argument("--output", required=True, help="Output Perfetto JSON path.")
    parser.add_argument(
        "--mode",
        choices=("full", "active_counters"),
        default="full",
        help="Export full request timeline or sampled active request counters.",
    )
    parser.add_argument(
        "--sample-interval-ms",
        type=float,
        default=100.0,
        help="Sampling interval for --mode active_counters.",
    )
    parser.add_argument(
        "--counter-scope",
        choices=("global", "process", "both"),
        default="both",
        help="Deprecated compatibility flag; active_counters uses agent/server/engine groups.",
    )
    args = parser.parse_args()

    records = _iter_records(Path(args.input))
    if args.mode == "full":
        trace = records_to_perfetto(records)
    else:
        trace = records_to_active_counter_perfetto(
            records,
            sample_interval_ms=args.sample_interval_ms,
            counter_scope=args.counter_scope,
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(trace, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
