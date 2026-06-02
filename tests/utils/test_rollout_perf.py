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
import json

from verl.utils.rollout_perf.context import current_trace_context, push_trace_context
from verl.utils.rollout_perf.export_perfetto import records_to_perfetto
from verl.utils.rollout_perf.writer import RolloutPerfWriter


def test_push_trace_context_restores_previous_values():
    assert current_trace_context() == {}
    with push_trace_context(trajectory_id="traj-1", turn_index=0):
        assert current_trace_context()["trajectory_id"] == "traj-1"
        with push_trace_context(turn_index=1, tool_name="calculator"):
            context = current_trace_context()
            assert context["trajectory_id"] == "traj-1"
            assert context["turn_index"] == 1
            assert context["tool_name"] == "calculator"
        assert current_trace_context()["turn_index"] == 0
    assert current_trace_context() == {}


def test_rollout_perf_writer_writes_jsonl_record(tmp_path):
    writer = RolloutPerfWriter(str(tmp_path), "unit_test", flush_interval_s=0, run_id="run")
    writer.write({"record_type": "event", "name": "sample", "payload": {"value": 1}})
    writer.close()

    files = list(tmp_path.glob("rollout_perf_run_unit_test_*.jsonl"))
    assert len(files) == 1
    record = json.loads(files[0].read_text().strip())
    assert record["schema_version"] == "rollout_perf.v1"
    assert record["record_type"] == "event"
    assert record["name"] == "sample"
    assert record["payload"]["value"] == 1
    assert record["role"] == "unit_test"


def test_records_to_perfetto_exports_span_and_counter():
    records = [
        {
            "record_type": "span",
            "name": "llm_turn",
            "role": "agent_loop_worker",
            "hostname": "host",
            "pid": 1,
            "start_unix_ns": 1_000_000,
            "duration_ns": 2_000_000,
            "context": {"trajectory_id": "traj-1"},
            "payload": {"output_token_count": 8},
        },
        {
            "record_type": "counter",
            "name": "inflight_requests",
            "role": "vllm_server",
            "hostname": "host",
            "pid": 2,
            "time_unix_ns": 2_000_000,
            "context": {"server_id": "server-0"},
            "payload": {"value": 3},
        },
    ]

    trace = records_to_perfetto(records)
    events = trace["traceEvents"]
    phases = {event["ph"] for event in events}
    assert "X" in phases
    assert "C" in phases
    assert any(event.get("name") == "llm_turn" and event.get("args", {}).get("output_token_count") == 8 for event in events)
