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


def main() -> None:
    parser = argparse.ArgumentParser(description="Export rollout perf JSONL records to Perfetto JSON.")
    parser.add_argument("--input", required=True, help="Input rollout perf JSONL file or directory.")
    parser.add_argument("--output", required=True, help="Output Perfetto JSON path.")
    args = parser.parse_args()

    trace = records_to_perfetto(_iter_records(Path(args.input)))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(trace, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
