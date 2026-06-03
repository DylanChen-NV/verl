# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from trace_loader import iter_records, payload, span_bounds
else:
    from .trace_loader import iter_records, payload, span_bounds


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((percentile / 100.0) * (len(ordered) - 1)))
    return float(ordered[max(0, min(index, len(ordered) - 1))])


def summarize_trace(input_path: str | Path) -> dict[str, Any]:
    counts = Counter()
    llm_latencies_ms = []
    engine_latencies_ms = []
    trajectory_starts = []
    trajectory_ends = []
    totals = Counter()
    for record in iter_records(input_path):
        record_type = record.get("record_type")
        name = record.get("name")
        counts[f"{record_type}/{name}"] += 1
        if record_type != "span":
            continue
        bounds = span_bounds(record)
        record_payload = payload(record)
        duration_ms = float(record.get("duration_ns") or 0) / 1_000_000
        if name == "trajectory":
            totals["trajectories"] += 1
            if bounds is not None:
                trajectory_starts.append(bounds[0])
                trajectory_ends.append(bounds[1])
        elif name == "llm_turn":
            totals["llm_turns"] += 1
            llm_latencies_ms.append(duration_ms)
            totals["llm_generated_tokens"] += int(record_payload.get("output_token_count") or 0)
        elif name == "vllm_engine_request":
            totals["engine_requests"] += 1
            engine_latencies_ms.append(duration_ms)
            totals["engine_prompt_tokens"] += int(record_payload.get("prompt_token_count") or 0)
            totals["engine_output_tokens"] += int(record_payload.get("output_token_count") or 0)
        elif name == "tool_call":
            totals["tool_calls"] += 1
            totals["tool_sleep_ms"] += duration_ms
            totals["tool_output_tokens"] += int(record_payload.get("output_token_count") or 0)

    wall_time_ms = 0.0
    if trajectory_starts and trajectory_ends:
        wall_time_ms = (max(trajectory_ends) - min(trajectory_starts)) / 1_000_000
    return {
        "input": str(input_path),
        "record_counts": dict(sorted(counts.items())),
        "totals": dict(totals),
        "wall_time_ms": wall_time_ms,
        "llm_latency_ms": {
            "mean": mean(llm_latencies_ms) if llm_latencies_ms else 0.0,
            "p50": _percentile(llm_latencies_ms, 50),
            "p95": _percentile(llm_latencies_ms, 95),
            "p99": _percentile(llm_latencies_ms, 99),
        },
        "engine_latency_ms": {
            "mean": mean(engine_latencies_ms) if engine_latencies_ms else 0.0,
            "p50": _percentile(engine_latencies_ms, 50),
            "p95": _percentile(engine_latencies_ms, 95),
            "p99": _percentile(engine_latencies_ms, 99),
        },
    }


def _relative_delta(original: float, replay: float) -> float | None:
    if original == 0:
        return None if replay != 0 else 0.0
    return (replay - original) / original


def compare_summaries(original: dict[str, Any], replay: dict[str, Any]) -> dict[str, Any]:
    metrics = [
        ("trajectories", original["totals"].get("trajectories", 0), replay["totals"].get("trajectories", 0)),
        ("engine_requests", original["totals"].get("engine_requests", 0), replay["totals"].get("engine_requests", 0)),
        ("engine_output_tokens", original["totals"].get("engine_output_tokens", 0), replay["totals"].get("engine_output_tokens", 0)),
        ("tool_calls", original["totals"].get("tool_calls", 0), replay["totals"].get("tool_calls", 0)),
        ("tool_sleep_ms", original["totals"].get("tool_sleep_ms", 0.0), replay["totals"].get("tool_sleep_ms", 0.0)),
        ("wall_time_ms", original["wall_time_ms"], replay["wall_time_ms"]),
    ]
    rows = []
    for name, original_value, replay_value in metrics:
        rows.append(
            {
                "metric": name,
                "original": original_value,
                "replay": replay_value,
                "relative_delta": _relative_delta(float(original_value), float(replay_value)),
            }
        )
    return {"original": original, "replay": replay, "metrics": rows}


def _write_markdown(report: dict[str, Any], output: Path) -> None:
    lines = [
        "# Rollout Perf Replay Compare",
        "",
        "| Metric | Original | Replay | Relative Delta |",
        "|---|---:|---:|---:|",
    ]
    for row in report["metrics"]:
        delta = row["relative_delta"]
        delta_text = "n/a" if delta is None else f"{delta:.3f}"
        lines.append(f"| {row['metric']} | {row['original']} | {row['replay']} | {delta_text} |")
    lines.extend(
        [
            "",
            "## Latency",
            "",
            "```json",
            json.dumps(
                {
                    "original_llm_latency_ms": report["original"]["llm_latency_ms"],
                    "replay_llm_latency_ms": report["replay"]["llm_latency_ms"],
                    "original_engine_latency_ms": report["original"]["engine_latency_ms"],
                    "replay_engine_latency_ms": report["replay"]["engine_latency_ms"],
                },
                indent=2,
                sort_keys=True,
            ),
            "```",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare original and replay rollout perf raw traces.")
    parser.add_argument("--original", required=True)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--output", required=True, help="Markdown report path.")
    parser.add_argument("--json-output", default=None, help="Optional JSON report path.")
    args = parser.parse_args()

    report = compare_summaries(summarize_trace(args.original), summarize_trace(args.replay))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_markdown(report, output)
    if args.json_output:
        json_output = Path(args.json_output)
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "metrics": report["metrics"]}, sort_keys=True))


if __name__ == "__main__":
    main()
