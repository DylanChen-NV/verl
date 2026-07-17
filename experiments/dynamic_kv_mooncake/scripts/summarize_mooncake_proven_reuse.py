#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def percentile(values, ratio):
    values = sorted(values)
    return values[int((len(values) - 1) * ratio)] if values else None


def distribution(values):
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "mean": sum(values) / len(values) if values else None,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values) if values else None,
    }


def interval_union_ms(records):
    intervals = sorted((row["scheduled_ns"], row["first_output_ns"]) for row in records)
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(end - start for start, end in merged) / 1_000_000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("strict_json", type=Path)
    parser.add_argument("queue_json", type=Path)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()

    strict = json.loads(args.strict_json.read_text())
    queue = json.loads(args.queue_json.read_text())
    attempts = {row["engine_request_id"]: row for row in queue["attempts"]}
    rows = []
    for proof in strict["requests"]:
        if not proof["strict_cross_node_reuse"]:
            continue
        attempt = attempts.get(proof["retry_engine_request_id"])
        if not attempt or attempt.get("status") != "completed":
            continue
        prefix = int(attempt["retry_prefix_tokens"])
        hit = int(proof["recv_hit_tokens"])
        rows.append(
            {
                **proof,
                "retry_prefix_tokens": prefix,
                "initial_computed_tokens": attempt["initial_computed_tokens"],
                "num_scheduled_tokens": attempt["num_scheduled_tokens"],
                "queue_time_ms": attempt["queue_time_ms"],
                "queue_excluded_time_ms": attempt["queue_excluded_time_ms"],
                "scheduled_ns": attempt["scheduled_ns"],
                "first_output_ns": attempt["first_output_ns"],
                "hit_ratio": hit / prefix,
            }
        )
    result = {
        "proven_completed_request_count": len(rows),
        "hit_ratio": distribution([row["hit_ratio"] for row in rows]),
        "recv_hit_tokens": distribution([row["recv_hit_tokens"] for row in rows]),
        "num_scheduled_tokens": distribution([row["num_scheduled_tokens"] for row in rows]),
        "queue_time_ms": distribution([row["queue_time_ms"] for row in rows]),
        "queue_excluded_time_ms": distribution([row["queue_excluded_time_ms"] for row in rows]),
        "queue_excluded_union_ms": interval_union_ms(rows),
        "requests": rows,
    }
    args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "requests"}, indent=2))


if __name__ == "__main__":
    main()
