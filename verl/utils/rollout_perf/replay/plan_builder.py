# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from schema import ReplayLLMRequest, ReplayPlan, ReplayToolCall, ReplayTrajectory, ReplayTurn
    from trace_loader import int_or_none, load_records, merged_context, non_negative_int, payload, span_bounds
else:
    from .schema import ReplayLLMRequest, ReplayPlan, ReplayToolCall, ReplayTrajectory, ReplayTurn
    from .trace_loader import int_or_none, load_records, merged_context, non_negative_int, payload, span_bounds


def _span_key(record: dict[str, Any]) -> tuple[str, int] | None:
    context = merged_context(record)
    trajectory_id = context.get("trajectory_id")
    turn_index = int_or_none(context.get("turn_index"))
    if trajectory_id is None or turn_index is None:
        return None
    return str(trajectory_id), turn_index


def _first(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not records:
        return None
    return sorted(records, key=lambda item: int(item.get("start_unix_ns") or item.get("time_unix_ns") or 0))[0]


def build_replay_plan(
    records: list[dict[str, Any]],
    *,
    source: str | None = None,
    max_trajectories: int | None = None,
) -> ReplayPlan:
    trajectory_spans: dict[str, dict[str, Any]] = {}
    llm_turns: dict[tuple[str, int], list[dict[str, Any]]] = {}
    llm_clients: dict[tuple[str, int], list[dict[str, Any]]] = {}
    tool_turns: dict[tuple[str, int], list[dict[str, Any]]] = {}
    tool_calls: dict[tuple[str, int], list[dict[str, Any]]] = {}
    engine_by_request_id: dict[str, dict[str, Any]] = {}
    engine_by_turn: dict[tuple[str, int], list[dict[str, Any]]] = {}

    min_trajectory_start = None
    for record in records:
        if record.get("record_type") != "span":
            continue
        name = record.get("name")
        bounds = span_bounds(record)
        if bounds is None:
            continue
        if name == "trajectory":
            context = merged_context(record)
            trajectory_id = context.get("trajectory_id")
            if trajectory_id is None:
                continue
            trajectory_id = str(trajectory_id)
            trajectory_spans[trajectory_id] = record
            min_trajectory_start = bounds[0] if min_trajectory_start is None else min(min_trajectory_start, bounds[0])
            continue

        key = _span_key(record)
        if key is None:
            continue
        if name == "llm_turn":
            llm_turns.setdefault(key, []).append(record)
        elif name == "llm_client_request":
            llm_clients.setdefault(key, []).append(record)
        elif name == "tool_turn":
            tool_turns.setdefault(key, []).append(record)
        elif name == "tool_call":
            tool_calls.setdefault(key, []).append(record)
        elif name == "vllm_engine_request":
            engine_by_turn.setdefault(key, []).append(record)
            engine_request_id = merged_context(record).get("engine_request_id")
            if engine_request_id is not None:
                engine_by_request_id[str(engine_request_id)] = record

    if min_trajectory_start is None:
        min_trajectory_start = 0

    trajectories: list[ReplayTrajectory] = []
    sorted_trajectory_items = sorted(
        trajectory_spans.items(), key=lambda item: span_bounds(item[1])[0] if span_bounds(item[1]) else 0
    )
    if max_trajectories is not None:
        sorted_trajectory_items = sorted_trajectory_items[:max_trajectories]

    for trajectory_id, trajectory_record in sorted_trajectory_items:
        trajectory_bounds = span_bounds(trajectory_record)
        if trajectory_bounds is None:
            continue
        trajectory_context = merged_context(trajectory_record)
        trajectory_payload = payload(trajectory_record)
        turn_indices = sorted(turn_index for tid, turn_index in llm_turns if tid == trajectory_id)
        turns: list[ReplayTurn] = []
        for turn_index in turn_indices:
            key = (trajectory_id, turn_index)
            llm_turn = _first(llm_turns.get(key, []))
            if llm_turn is None:
                continue
            llm_payload = payload(llm_turn)
            llm_context = merged_context(llm_turn)
            client = _first(llm_clients.get(key, []))
            client_payload = payload(client) if client else {}
            client_context = merged_context(client) if client else {}
            engine_request_id = client_payload.get("engine_request_id") or client_context.get("engine_request_id")
            engine = engine_by_request_id.get(str(engine_request_id)) if engine_request_id is not None else None
            if engine is None:
                engine = _first(engine_by_turn.get(key, []))
            engine_payload = payload(engine) if engine else {}

            prompt_token_count = non_negative_int(
                client_payload.get("prompt_token_count", engine_payload.get("prompt_token_count", llm_payload.get("input_token_count")))
            )
            output_token_count = non_negative_int(
                client_payload.get("output_token_count", engine_payload.get("output_token_count", llm_payload.get("output_token_count")))
            )
            llm_request = ReplayLLMRequest(
                logical_request_id=(
                    str(client_payload.get("logical_request_id") or client_context.get("logical_request_id"))
                    if (client_payload.get("logical_request_id") or client_context.get("logical_request_id")) is not None
                    else None
                ),
                engine_request_id=str(engine_request_id) if engine_request_id is not None else None,
                prompt_token_count=prompt_token_count,
                output_token_count=output_token_count,
                requested_max_tokens=int_or_none(
                    client_payload.get("requested_max_tokens", llm_payload.get("requested_max_tokens"))
                ),
                original_llm_duration_ns=int_or_none(llm_turn.get("duration_ns")),
                original_client_duration_ns=int_or_none(client.get("duration_ns")) if client else None,
                original_engine_duration_ns=int_or_none(engine.get("duration_ns")) if engine else None,
                original_ttft_ns=int_or_none(engine_payload.get("ttft_ns")),
                stop_reason=client_payload.get("stop_reason") or engine_payload.get("stop_reason") or llm_payload.get("stop_reason"),
                status=str(llm_turn.get("status") or "ok"),
            )

            turn_tool_calls = []
            for tool_record in sorted(tool_calls.get(key, []), key=lambda item: int(item.get("start_unix_ns") or 0)):
                tool_payload = payload(tool_record)
                tool_context = merged_context(tool_record)
                turn_tool_calls.append(
                    ReplayToolCall(
                        tool_call_id=(
                            str(tool_context.get("tool_call_id")) if tool_context.get("tool_call_id") is not None else None
                        ),
                        tool_name=(
                            str(tool_payload.get("tool_name") or tool_context.get("tool_name"))
                            if (tool_payload.get("tool_name") or tool_context.get("tool_name")) is not None
                            else None
                        ),
                        argument_token_count=non_negative_int(tool_payload.get("argument_token_count")),
                        output_token_count=non_negative_int(tool_payload.get("output_token_count")),
                        sleep_ns=non_negative_int(tool_record.get("duration_ns")),
                        status=str(tool_record.get("status") or "ok"),
                    )
                )
            tool_turn = _first(tool_turns.get(key, []))
            tool_turn_payload = payload(tool_turn) if tool_turn else {}
            turns.append(
                ReplayTurn(
                    turn_index=turn_index,
                    llm_request=llm_request,
                    tool_calls=turn_tool_calls,
                    tool_turn_duration_ns=int_or_none(tool_turn.get("duration_ns")) if tool_turn else None,
                    tool_response_token_count=non_negative_int(tool_turn_payload.get("tool_response_token_count")),
                )
            )

        trajectories.append(
            ReplayTrajectory(
                trajectory_id=trajectory_id,
                global_step=int_or_none(trajectory_context.get("global_step")),
                sample_index=int_or_none(trajectory_context.get("sample_index")),
                rollout_n=int_or_none(trajectory_context.get("rollout_n")),
                validate=trajectory_context.get("validate"),
                agent_name=trajectory_context.get("agent_name"),
                start_offset_ns=trajectory_bounds[0] - min_trajectory_start,
                original_duration_ns=max(0, trajectory_bounds[1] - trajectory_bounds[0]),
                prompt_token_count=non_negative_int(trajectory_payload.get("prompt_token_count")),
                response_token_count=non_negative_int(trajectory_payload.get("response_token_count")),
                llm_generated_token_count=non_negative_int(trajectory_payload.get("llm_generated_token_count")),
                tool_response_token_count=non_negative_int(trajectory_payload.get("tool_response_token_count")),
                tool_call_count=non_negative_int(trajectory_payload.get("tool_call_count")),
                turns=turns,
            )
        )

    return ReplayPlan(
        metadata={
            "source": source,
            "source_record_count": len(records),
            "time_origin_unix_ns": min_trajectory_start,
            "built_unix_ns": time.time_ns(),
            "max_trajectories": max_trajectories,
        },
        trajectories=trajectories,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a closed-loop replay plan from rollout perf raw trace.")
    parser.add_argument("--input", required=True, help="Input raw trace JSONL file or directory.")
    parser.add_argument("--output", required=True, help="Output replay plan JSON path.")
    parser.add_argument("--max-trajectories", type=int, default=None, help="Optional first-N trajectory limit.")
    args = parser.parse_args()

    records = load_records(args.input)
    plan = build_replay_plan(records, source=args.input, max_trajectories=args.max_trajectories)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(plan.summary(), sort_keys=True))


if __name__ == "__main__":
    main()
