# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def iter_records(input_path: str | Path) -> Iterable[dict[str, Any]]:
    path = Path(input_path)
    paths = [path]
    if path.is_dir():
        paths = sorted(path.glob("rollout_perf_*.jsonl"))
    for record_path in paths:
        with record_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def load_records(input_path: str | Path) -> list[dict[str, Any]]:
    return list(iter_records(input_path))


def span_bounds(record: dict[str, Any]) -> tuple[int, int] | None:
    start = record.get("start_unix_ns")
    end = record.get("end_unix_ns")
    if start is None:
        return None
    if end is None:
        duration = record.get("duration_ns")
        if duration is None:
            return None
        end = int(start) + int(duration)
    if int(end) < int(start):
        return None
    return int(start), int(end)


def merged_context(record: dict[str, Any]) -> dict[str, Any]:
    context = dict(record.get("context") or {})
    payload = record.get("payload") or {}
    for key, value in payload.items():
        context.setdefault(key, value)
    return context


def payload(record: dict[str, Any]) -> dict[str, Any]:
    return dict(record.get("payload") or {})


def int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def non_negative_int(value: Any, default: int = 0) -> int:
    parsed = int_or_none(value)
    if parsed is None:
        return default
    return max(0, parsed)
