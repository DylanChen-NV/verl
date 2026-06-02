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
from collections import Counter
from typing import Any, Iterable


def summarize_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Return a compact summary for rollout perf records."""
    counts = Counter()
    span_duration_ns = Counter()
    for record in records:
        name = record.get("name", "unknown")
        counts[(record.get("record_type", "unknown"), name)] += 1
        if record.get("record_type") == "span":
            span_duration_ns[name] += int(record.get("duration_ns") or 0)
    return {
        "record_counts": {f"{kind}/{name}": count for (kind, name), count in counts.items()},
        "span_duration_ms_sum": {name: value / 1_000_000 for name, value in span_duration_ns.items()},
    }
