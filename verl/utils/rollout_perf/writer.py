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
import atexit
import dataclasses
import json
import os
import socket
import threading
import time
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4


def _json_default(value: Any):
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    return repr(value)


class RolloutPerfWriter:
    """Process-local JSONL writer for rollout perf trace records."""

    def __init__(self, output_dir: str, role: str, *, flush_interval_s: float = 5.0, run_id: Optional[str] = None):
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.role = role
        self.flush_interval_s = flush_interval_s
        self.hostname = socket.gethostname()
        self.pid = os.getpid()
        self.run_id = run_id or os.getenv("VERL_ROLLOUT_PERF_RUN_ID") or uuid4().hex[:12]
        filename = f"rollout_perf_{self.run_id}_{role}_{self.hostname}_{self.pid}_{uuid4().hex[:8]}.jsonl"
        self.path = self.output_dir / filename
        self._lock = threading.Lock()
        self._last_flush = time.monotonic()
        self._file = self.path.open("a", encoding="utf-8", buffering=1)
        atexit.register(self.close)

    def write(self, record: dict[str, Any]) -> None:
        now_ns = time.time_ns()
        monotonic_ns = time.monotonic_ns()
        enriched = {
            "schema_version": "rollout_perf.v1",
            "record_id": uuid4().hex,
            "time_unix_ns": now_ns,
            "time_monotonic_ns": monotonic_ns,
            "hostname": self.hostname,
            "pid": self.pid,
            "role": self.role,
            "run_id": self.run_id,
            **record,
        }
        line = json.dumps(enriched, default=_json_default, ensure_ascii=False, sort_keys=True)
        with self._lock:
            self._file.write(line + "\n")
            if self.flush_interval_s == 0 or time.monotonic() - self._last_flush >= self.flush_interval_s:
                self._file.flush()
                self._last_flush = time.monotonic()

    def close(self) -> None:
        with self._lock:
            if not self._file.closed:
                self._file.flush()
                self._file.close()


_WRITER: Optional[RolloutPerfWriter] = None
_ENABLED = False


def init_writer(config: Any, role: str) -> Optional[RolloutPerfWriter]:
    """Initialize the process-local writer from a dataclass/DictConfig/dict config."""
    global _WRITER, _ENABLED
    enabled = bool(_config_get(config, "enable", False))
    _ENABLED = enabled
    if not enabled:
        return None
    if _WRITER is not None:
        return _WRITER
    output_dir = _config_get(config, "output_dir", None) or os.getenv("VERL_ROLLOUT_PERF_OUTPUT_DIR")
    if not output_dir:
        output_dir = os.path.abspath("rollout_perf_trace")
    flush_interval_s = float(_config_get(config, "flush_interval_s", 5.0))
    _WRITER = RolloutPerfWriter(output_dir, role, flush_interval_s=flush_interval_s)
    _WRITER.write(
        {
            "record_type": "run_manifest",
            "name": "writer_started",
            "payload": {
                "output_dir": str(_WRITER.output_dir),
                "trace_file": str(_WRITER.path),
                "format": _config_get(config, "format", "jsonl"),
                "capture_content": bool(_config_get(config, "capture_content", False)),
            },
        }
    )
    return _WRITER


def get_writer() -> Optional[RolloutPerfWriter]:
    return _WRITER


def writer_enabled() -> bool:
    return _ENABLED and _WRITER is not None


def write_record(record: dict[str, Any]) -> None:
    if writer_enabled():
        assert _WRITER is not None
        _WRITER.write(record)


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)
