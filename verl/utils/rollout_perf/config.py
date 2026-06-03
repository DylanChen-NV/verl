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
from dataclasses import dataclass, field
from typing import Optional

from verl.base_config import BaseConfig


@dataclass
class PerfTraceEngineInternalConfig(BaseConfig):
    """Engine-internal rollout performance trace settings."""

    enable: bool = False
    backend: str = "auto"
    sample_interval_ms: int = 500


@dataclass
class PerfTraceConfig(BaseConfig):
    """Rollout performance trace settings.

    This config is intentionally separate from rollout.trace, which targets
    external trace backends such as weave/mlflow/trackio.
    """

    enable: bool = False
    output_dir: Optional[str] = None
    format: str = "jsonl"
    capture_content: bool = False
    max_samples_per_step_per_worker: Optional[int] = None
    flush_interval_s: float = 5.0
    export_perfetto: bool = False
    engine_internal: PerfTraceEngineInternalConfig = field(default_factory=PerfTraceEngineInternalConfig)

    def __post_init__(self):
        if self.format not in {"jsonl"}:
            raise ValueError(f"Unsupported rollout perf trace format: {self.format}")
        if self.max_samples_per_step_per_worker is not None and self.max_samples_per_step_per_worker < 0:
            raise ValueError("`max_samples_per_step_per_worker` must be a non-negative integer or null.")
        if self.flush_interval_s < 0:
            raise ValueError("`flush_interval_s` must be non-negative.")
