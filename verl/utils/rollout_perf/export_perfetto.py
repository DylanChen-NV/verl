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
from pathlib import Path
from typing import Any, Iterable


ACTIVE_COUNTER_SPANS = {
    "trajectory",
    "llm_turn",
    "llm_client_request",
    "vllm_engine_request",
    "tool_turn",
    "tool_call",
}


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


def _counter_process_name(scope: str, key: str) -> str:
    if scope == "global":
        return "rollout_perf_active@global"
    return f"rollout_perf_active@{key}"


def _add_counter_metadata(events: list[dict[str, Any]], process_ids: dict[str, int], process: str) -> int:
    if process not in process_ids:
        process_ids[process] = len(process_ids) + 1
        pid = process_ids[process]
        events.append({"name": "process_name", "ph": "M", "pid": pid, "args": {"name": process}})
        events.append({"name": "thread_name", "ph": "M", "pid": pid, "tid": 1, "args": {"name": "active_counters"}})
    return process_ids[process]


def _process_span_groups(
    records: Iterable[dict[str, Any]],
    counter_scope: str,
) -> dict[tuple[str, str, str], list[tuple[int, int]]]:
    groups: dict[tuple[str, str, str], list[tuple[int, int]]] = {}
    include_global = counter_scope in {"global", "both"}
    include_process = counter_scope in {"process", "both"}

    for record in records:
        if record.get("record_type") != "span":
            continue
        span_name = record.get("name")
        if span_name not in ACTIVE_COUNTER_SPANS:
            continue
        bounds = _span_bounds(record)
        if bounds is None:
            continue
        if include_global:
            groups.setdefault(("global", "all", span_name), []).append(bounds)
        if include_process:
            groups.setdefault(("process", _process_name(record), span_name), []).append(bounds)
    return groups


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


def records_to_active_counter_perfetto(
    records: Iterable[dict[str, Any]],
    *,
    sample_interval_ms: float,
    counter_scope: str,
) -> dict[str, Any]:
    if sample_interval_ms <= 0:
        raise ValueError("--sample-interval-ms must be positive")
    sample_interval_ns = int(sample_interval_ms * 1_000_000)
    if sample_interval_ns <= 0:
        raise ValueError("--sample-interval-ms is too small")

    groups = _process_span_groups(records, counter_scope)
    events: list[dict[str, Any]] = []
    process_ids: dict[str, int] = {}

    for (scope, group_key, span_name), spans in sorted(groups.items()):
        process = _counter_process_name(scope, group_key)
        pid = _add_counter_metadata(events, process_ids, process)
        for sample_ts, active in _iter_counter_samples(spans, sample_interval_ns):
            args = {
                "value": active,
                "span_name": span_name,
                "scope": scope,
            }
            if scope == "process":
                args["process"] = group_key
            events.append(
                {
                    "name": f"active/{span_name}",
                    "cat": "rollout_perf_active",
                    "ph": "C",
                    "ts": sample_ts / 1000,
                    "pid": pid,
                    "tid": 1,
                    "args": args,
                }
            )
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
        help="Counter aggregation scope for --mode active_counters.",
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
