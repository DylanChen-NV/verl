# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
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


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _value_stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"samples": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0, "sum": 0.0}
    return {
        "samples": len(values),
        "mean": mean(values),
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
        "max": max(values),
        "sum": sum(values),
    }


def summarize_trace(input_path: str | Path) -> dict[str, Any]:
    counts = Counter()
    llm_latencies_ms = []
    engine_latencies_ms = []
    trajectory_starts = []
    trajectory_ends = []
    totals = Counter()
    counter_values: dict[str, list[float]] = defaultdict(list)
    counter_max_values: dict[str, list[float]] = defaultdict(list)

    for record in iter_records(input_path):
        record_type = record.get("record_type")
        name = record.get("name")
        counts[f"{record_type}/{name}"] += 1
        record_payload = payload(record)
        if record_type == "counter":
            value = _as_float(record_payload.get("value"))
            if value is not None and name:
                counter_values[str(name)].append(value)
            max_value = _as_float(record_payload.get("max_value"))
            if max_value is not None and name:
                counter_max_values[str(name)].append(max_value)
            continue
        if record_type != "span":
            continue
        bounds = span_bounds(record)
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
    counter_summary = {}
    for counter_name in sorted(set(counter_values) | set(counter_max_values)):
        counter_summary[counter_name] = {
            "value": _value_stats(counter_values.get(counter_name, [])),
            "max_value": _value_stats(counter_max_values.get(counter_name, [])),
        }
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
        "counter_summary": counter_summary,
    }


def _relative_delta(original: float, replay: float) -> float | None:
    if original == 0:
        return None if replay != 0 else 0.0
    return (replay - original) / original


def _counter_compare_rows(original: dict[str, Any], replay: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    original_counters = original.get("counter_summary", {})
    replay_counters = replay.get("counter_summary", {})
    for name in sorted(set(original_counters) | set(replay_counters)):
        original_value = original_counters.get(name, {}).get("value", _value_stats([]))
        replay_value = replay_counters.get(name, {}).get("value", _value_stats([]))
        original_mean = float(original_value.get("mean") or 0.0)
        replay_mean = float(replay_value.get("mean") or 0.0)
        rows.append(
            {
                "counter": name,
                "original": original_value,
                "replay": replay_value,
                "mean_relative_delta": _relative_delta(original_mean, replay_mean),
            }
        )
    return rows


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
    return {"original": original, "replay": replay, "metrics": rows, "counter_metrics": _counter_compare_rows(original, replay)}


def _format_number(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(numeric) >= 1000:
        return f"{numeric:.1f}"
    return f"{numeric:.3f}"


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
    if report.get("counter_metrics"):
        lines.extend(
            [
                "",
                "## Counter Summary",
                "",
                "`value` is summarized directly from raw counter records. If a counter has `payload.max_value`, its stats are included in the JSON report.",
                "",
                "| Counter | Orig Samples | Orig Mean | Orig P95 | Orig Max | Replay Samples | Replay Mean | Replay P95 | Replay Max | Mean Delta |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in report["counter_metrics"]:
            original = row["original"]
            replay = row["replay"]
            delta = row["mean_relative_delta"]
            delta_text = "n/a" if delta is None else f"{delta:.3f}"
            lines.append(
                "| {counter} | {os} | {om} | {op95} | {omax} | {rs} | {rm} | {rp95} | {rmax} | {delta} |".format(
                    counter=row["counter"],
                    os=original.get("samples", 0),
                    om=_format_number(original.get("mean")),
                    op95=_format_number(original.get("p95")),
                    omax=_format_number(original.get("max")),
                    rs=replay.get("samples", 0),
                    rm=_format_number(replay.get("mean")),
                    rp95=_format_number(replay.get("p95")),
                    rmax=_format_number(replay.get("max")),
                    delta=delta_text,
                )
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
