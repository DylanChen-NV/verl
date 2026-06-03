# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent / "engines"))
    from engines.mock import MockReplayEngine
    from replay_writer import ReplaySpan, ReplayTraceWriter
    from schema import ReplayPlan, ReplayToolCall, ReplayTrajectory, ReplayTurn
    from tool_mock import ToolSleepMock
else:
    from .engines.mock import MockReplayEngine
    from .replay_writer import ReplaySpan, ReplayTraceWriter
    from .schema import ReplayPlan, ReplayToolCall, ReplayTrajectory, ReplayTurn
    from .tool_mock import ToolSleepMock


def _load_plan(path: str | Path) -> ReplayPlan:
    with Path(path).open("r", encoding="utf-8") as f:
        return ReplayPlan.from_dict(json.load(f))


def _trajectory_context(trajectory: ReplayTrajectory) -> dict[str, Any]:
    return {
        "trace_sampled": True,
        "trajectory_id": trajectory.trajectory_id,
        "global_step": trajectory.global_step,
        "sample_index": trajectory.sample_index,
        "rollout_n": trajectory.rollout_n,
        "validate": trajectory.validate,
        "agent_name": trajectory.agent_name,
        "replay": True,
    }


def _turn_context(trajectory: ReplayTrajectory, turn: ReplayTurn, *, turn_kind: str) -> dict[str, Any]:
    context = _trajectory_context(trajectory)
    context.update(
        {
            "turn_index": turn.turn_index,
            "turn_kind": turn_kind,
            "logical_request_id": turn.llm_request.logical_request_id,
        }
    )
    return context


async def _run_tool_call(
    writer: ReplayTraceWriter,
    tool_mock: ToolSleepMock,
    trajectory: ReplayTrajectory,
    turn: ReplayTurn,
    tool_call: ReplayToolCall,
) -> dict[str, int | str | None]:
    context = _turn_context(trajectory, turn, turn_kind="tool")
    context.update({"tool_call_id": tool_call.tool_call_id, "tool_name": tool_call.tool_name})
    span = ReplaySpan(
        writer,
        role="agent_loop_worker",
        name="tool_call",
        context=context,
        payload={
            "tool_name": tool_call.tool_name,
            "argument_token_count": tool_call.argument_token_count,
            "output_token_count": tool_call.output_token_count,
            "replay_sleep_ns": tool_call.sleep_ns,
        },
    )
    try:
        result = await tool_mock.run(tool_call)
        span.finish(
            {
                "output_token_count": tool_call.output_token_count,
                "replay_sleep_ns": tool_call.sleep_ns,
            },
            status=tool_call.status,
        )
        return result
    except Exception:
        span.finish(status="error")
        raise


async def _run_turn(
    writer: ReplayTraceWriter,
    engine: MockReplayEngine,
    tool_mock: ToolSleepMock,
    trajectory: ReplayTrajectory,
    turn: ReplayTurn,
) -> None:
    llm_context = _turn_context(trajectory, turn, turn_kind="assistant")
    llm_request = turn.llm_request
    llm_turn_span = ReplaySpan(
        writer,
        role="agent_loop_worker",
        name="llm_turn",
        context=llm_context,
        payload={
            "input_token_count": llm_request.prompt_token_count,
            "requested_max_tokens": llm_request.requested_max_tokens,
            "target_output_token_count": llm_request.output_token_count,
        },
    )
    client_span = ReplaySpan(
        writer,
        role="agent_loop_worker",
        name="llm_client_request",
        context=llm_context,
        payload={
            "logical_request_id": llm_request.logical_request_id,
            "engine_request_id": llm_request.engine_request_id,
            "prompt_token_count": llm_request.prompt_token_count,
            "requested_max_tokens": llm_request.requested_max_tokens,
        },
    )
    engine_context = dict(llm_context)
    engine_context.update({"engine_backend": "mock", "engine_request_id": llm_request.engine_request_id})
    engine_span = ReplaySpan(
        writer,
        role="vllm_server",
        name="vllm_engine_request",
        context=engine_context,
        payload={
            "engine_request_id": llm_request.engine_request_id,
            "prompt_token_count": llm_request.prompt_token_count,
            "target_output_token_count": llm_request.output_token_count,
            "replay_engine": "mock",
        },
    )
    try:
        response = await engine.generate(llm_request)
        engine_span.finish(
            {
                "output_token_count": response.output_token_count,
                "stop_reason": response.stop_reason,
                "ttft_ns": response.ttft_ns,
            },
            status=llm_request.status,
        )
        client_span.finish(
            {
                "engine_request_id": llm_request.engine_request_id,
                "output_token_count": response.output_token_count,
                "prompt_token_count": llm_request.prompt_token_count,
                "requested_max_tokens": llm_request.requested_max_tokens,
                "stop_reason": response.stop_reason,
            },
            status=llm_request.status,
        )
        llm_turn_span.finish(
            {
                "output_token_count": response.output_token_count,
                "stop_reason": response.stop_reason,
            },
            status=llm_request.status,
        )
    except Exception:
        engine_span.finish(status="error")
        client_span.finish(status="error")
        llm_turn_span.finish(status="error")
        raise

    if turn.tool_calls:
        tool_context = _turn_context(trajectory, turn, turn_kind="tool")
        tool_turn_span = ReplaySpan(
            writer,
            role="agent_loop_worker",
            name="tool_turn",
            context=tool_context,
            payload={"tool_call_count": len(turn.tool_calls)},
        )
        try:
            await asyncio.gather(*(_run_tool_call(writer, tool_mock, trajectory, turn, tool) for tool in turn.tool_calls))
            tool_turn_span.finish(
                {
                    "tool_call_count": len(turn.tool_calls),
                    "tool_response_token_count": turn.tool_response_token_count
                    or sum(tool.output_token_count for tool in turn.tool_calls),
                }
            )
        except Exception:
            tool_turn_span.finish(status="error")
            raise


async def _run_trajectory(
    writer: ReplayTraceWriter,
    engine: MockReplayEngine,
    tool_mock: ToolSleepMock,
    trajectory: ReplayTrajectory,
    *,
    arrival_mode: str,
    time_scale: float,
) -> None:
    if arrival_mode == "original" and trajectory.start_offset_ns > 0 and time_scale > 0:
        await asyncio.sleep((trajectory.start_offset_ns / 1_000_000_000) * time_scale)
    span = ReplaySpan(
        writer,
        role="agent_loop_worker",
        name="trajectory",
        context=_trajectory_context(trajectory),
        payload={
            "agent_name": trajectory.agent_name,
            "sample_index": trajectory.sample_index,
            "rollout_n": trajectory.rollout_n,
            "prompt_token_count": trajectory.prompt_token_count,
        },
    )
    try:
        for turn in trajectory.turns:
            await _run_turn(writer, engine, tool_mock, trajectory, turn)
        span.finish(
            {
                "prompt_token_count": trajectory.prompt_token_count,
                "response_token_count": trajectory.response_token_count,
                "llm_generated_token_count": trajectory.llm_generated_token_count,
                "tool_response_token_count": trajectory.tool_response_token_count,
                "tool_call_count": trajectory.tool_call_count,
                "num_turns": len(trajectory.turns) + sum(1 for turn in trajectory.turns if turn.tool_calls),
            }
        )
    except Exception:
        span.finish(status="error")
        raise


async def run_replay(
    plan: ReplayPlan,
    *,
    output_dir: str | Path,
    engine_name: str,
    arrival_mode: str,
    time_scale: float,
    max_trajectories: int | None = None,
    max_concurrency: int | None = None,
    fixed_mock_latency_ms: float | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    if engine_name != "mock":
        raise ValueError("V0 P1 runner currently supports only --engine mock.")
    writer = ReplayTraceWriter(output_dir, run_id=run_id)
    engine = MockReplayEngine(time_scale=time_scale, fixed_latency_ms=fixed_mock_latency_ms)
    tool_mock = ToolSleepMock(time_scale=time_scale)
    trajectories = plan.trajectories[:max_trajectories] if max_trajectories is not None else plan.trajectories
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency and max_concurrency > 0 else None

    async def run_one(trajectory: ReplayTrajectory) -> None:
        if semaphore is None:
            await _run_trajectory(writer, engine, tool_mock, trajectory, arrival_mode=arrival_mode, time_scale=time_scale)
        else:
            async with semaphore:
                await _run_trajectory(writer, engine, tool_mock, trajectory, arrival_mode=arrival_mode, time_scale=time_scale)

    try:
        await asyncio.gather(*(run_one(trajectory) for trajectory in trajectories))
    finally:
        writer.close()

    return {
        "output_dir": str(Path(output_dir).resolve()),
        "trace_file": str(writer.path),
        "engine": engine_name,
        "arrival_mode": arrival_mode,
        "time_scale": time_scale,
        "trajectories": len(trajectories),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run closed-loop replay from a replay plan.")
    parser.add_argument("--plan", required=True, help="Replay plan JSON path.")
    parser.add_argument("--engine", choices=("mock",), default="mock", help="Replay engine.")
    parser.add_argument("--output-dir", required=True, help="Output raw replay trace directory.")
    parser.add_argument("--arrival-mode", choices=("original", "burst"), default="original")
    parser.add_argument("--time-scale", type=float, default=1.0, help="Scale original sleeps and arrival offsets.")
    parser.add_argument("--max-trajectories", type=int, default=None)
    parser.add_argument("--max-concurrency", type=int, default=None)
    parser.add_argument("--fixed-mock-latency-ms", type=float, default=None)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    plan = _load_plan(args.plan)
    summary = asyncio.run(
        run_replay(
            plan,
            output_dir=args.output_dir,
            engine_name=args.engine,
            arrival_mode=args.arrival_mode,
            time_scale=args.time_scale,
            max_trajectories=args.max_trajectories,
            max_concurrency=args.max_concurrency,
            fixed_mock_latency_ms=args.fixed_mock_latency_ms,
            run_id=args.run_id,
        )
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
