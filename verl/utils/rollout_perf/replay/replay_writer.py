# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
from __future__ import annotations

import dataclasses
import json
import os
import socket
import threading
import time
from pathlib import Path
from typing import Any
from uuid import uuid4


def _json_default(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return repr(value)


class ReplayTraceWriter:
    """Single-process JSONL writer for replay-generated rollout perf records."""

    def __init__(self, output_dir: str | Path, *, run_id: str | None = None):
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or f"replay-{uuid4().hex[:12]}"
        self.hostname = socket.gethostname()
        self.pid = os.getpid()
        self.path = self.output_dir / f"rollout_perf_{self.run_id}_replay_{self.hostname}_{self.pid}_{uuid4().hex[:8]}.jsonl"
        self._lock = threading.Lock()
        self._file = self.path.open("a", encoding="utf-8", buffering=1)
        self.write(
            role="replay",
            record_type="run_manifest",
            name="replay_writer_started",
            payload={"output_dir": str(self.output_dir), "trace_file": str(self.path)},
            context={},
        )

    def write(
        self,
        *,
        role: str,
        record_type: str,
        name: str,
        payload: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        status: str | None = None,
        **extra: Any,
    ) -> None:
        record = {
            "schema_version": "rollout_perf.replay.v1",
            "record_id": uuid4().hex,
            "time_unix_ns": time.time_ns(),
            "time_monotonic_ns": time.monotonic_ns(),
            "hostname": self.hostname,
            "pid": self.pid,
            "role": role,
            "run_id": self.run_id,
            "record_type": record_type,
            "name": name,
            "priority": "P0",
            "context": context or {},
            "payload": payload or {},
            **extra,
        }
        if status is not None:
            record["status"] = status
        line = json.dumps(record, default=_json_default, ensure_ascii=False, sort_keys=True)
        with self._lock:
            self._file.write(line + "\n")

    def close(self) -> None:
        with self._lock:
            if not self._file.closed:
                self._file.flush()
                self._file.close()


class ReplaySpan:
    def __init__(
        self,
        writer: ReplayTraceWriter,
        *,
        role: str,
        name: str,
        context: dict[str, Any],
        payload: dict[str, Any] | None = None,
    ):
        self.writer = writer
        self.role = role
        self.name = name
        self.context = dict(context)
        self.payload = dict(payload or {})
        self.start_unix_ns = time.time_ns()
        self.start_monotonic_ns = time.monotonic_ns()
        self._finished = False

    def finish(self, payload: dict[str, Any] | None = None, *, status: str = "ok") -> None:
        if self._finished:
            return
        self._finished = True
        end_unix_ns = time.time_ns()
        end_monotonic_ns = time.monotonic_ns()
        merged_payload = dict(self.payload)
        if payload:
            merged_payload.update(payload)
        self.writer.write(
            role=self.role,
            record_type="span",
            name=self.name,
            payload=merged_payload,
            context=self.context,
            status=status,
            start_unix_ns=self.start_unix_ns,
            end_unix_ns=end_unix_ns,
            start_monotonic_ns=self.start_monotonic_ns,
            end_monotonic_ns=end_monotonic_ns,
            duration_ns=end_monotonic_ns - self.start_monotonic_ns,
        )

    def __enter__(self) -> "ReplaySpan":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.finish(status="error" if exc is not None else "ok")
        return False
