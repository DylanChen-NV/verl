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
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


REPLAY_PLAN_VERSION = "rollout_perf_replay_plan.v1"


@dataclass
class ReplayLLMRequest:
    logical_request_id: str | None
    engine_request_id: str | None
    prompt_token_count: int
    output_token_count: int
    requested_max_tokens: int | None = None
    original_llm_duration_ns: int | None = None
    original_client_duration_ns: int | None = None
    original_engine_duration_ns: int | None = None
    original_ttft_ns: int | None = None
    stop_reason: str | None = None
    status: str = "ok"


@dataclass
class ReplayToolCall:
    tool_call_id: str | None
    tool_name: str | None
    argument_token_count: int
    output_token_count: int
    sleep_ns: int
    status: str = "ok"


@dataclass
class ReplayTurn:
    turn_index: int
    llm_request: ReplayLLMRequest
    tool_calls: list[ReplayToolCall] = field(default_factory=list)
    tool_turn_duration_ns: int | None = None
    tool_response_token_count: int = 0


@dataclass
class ReplayTrajectory:
    trajectory_id: str
    global_step: int | None
    sample_index: int | None
    rollout_n: int | None
    validate: bool | None
    agent_name: str | None
    start_offset_ns: int
    original_duration_ns: int
    prompt_token_count: int = 0
    response_token_count: int = 0
    llm_generated_token_count: int = 0
    tool_response_token_count: int = 0
    tool_call_count: int = 0
    turns: list[ReplayTurn] = field(default_factory=list)


@dataclass
class ReplayPlan:
    version: str = REPLAY_PLAN_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)
    trajectories: list[ReplayTrajectory] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReplayPlan":
        trajectories = []
        for trajectory_data in data.get("trajectories", []):
            turns = []
            for turn_data in trajectory_data.get("turns", []):
                llm_request = ReplayLLMRequest(**turn_data["llm_request"])
                tool_calls = [ReplayToolCall(**item) for item in turn_data.get("tool_calls", [])]
                turns.append(
                    ReplayTurn(
                        turn_index=turn_data["turn_index"],
                        llm_request=llm_request,
                        tool_calls=tool_calls,
                        tool_turn_duration_ns=turn_data.get("tool_turn_duration_ns"),
                        tool_response_token_count=int(turn_data.get("tool_response_token_count") or 0),
                    )
                )
            trajectory_kwargs = dict(trajectory_data)
            trajectory_kwargs["turns"] = turns
            trajectories.append(ReplayTrajectory(**trajectory_kwargs))
        return cls(
            version=data.get("version", REPLAY_PLAN_VERSION),
            metadata=dict(data.get("metadata") or {}),
            trajectories=trajectories,
        )

    def summary(self) -> dict[str, Any]:
        llm_requests = sum(len(trajectory.turns) for trajectory in self.trajectories)
        tool_calls = sum(len(turn.tool_calls) for trajectory in self.trajectories for turn in trajectory.turns)
        generated_tokens = sum(
            turn.llm_request.output_token_count for trajectory in self.trajectories for turn in trajectory.turns
        )
        tool_output_tokens = sum(
            tool.output_token_count
            for trajectory in self.trajectories
            for turn in trajectory.turns
            for tool in turn.tool_calls
        )
        return {
            "version": self.version,
            "trajectories": len(self.trajectories),
            "llm_requests": llm_requests,
            "tool_calls": tool_calls,
            "generated_tokens": generated_tokens,
            "tool_output_tokens": tool_output_tokens,
        }
