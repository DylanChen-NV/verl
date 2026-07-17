#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def summarize(run_dir):
    strict = json.loads((run_dir / "analysis/strict_cross_node_reuse_report.json").read_text())
    queue = json.loads((run_dir / "analysis/queue_excluded_report.json").read_text())
    attempts = {row["engine_request_id"]: row for row in queue["attempts"]}
    rows = [
        attempts[proof["retry_engine_request_id"]]
        for proof in strict["requests"]
        if proof["retry_cross_node"]
        and proof["retry_engine_request_id"] in attempts
        and attempts[proof["retry_engine_request_id"]].get("status") == "completed"
    ]
    intervals = sorted((row["scheduled_ns"], row["first_output_ns"]) for row in rows)
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    values = sorted(row["queue_excluded_time_ms"] for row in rows)
    return {
        "request_count": len(rows),
        "mean_ms": sum(values) / len(values),
        "p50_ms": values[(len(values) - 1) // 2],
        "min_ms": min(values),
        "max_ms": max(values),
        "union_ms": sum(end - start for start, end in merged) / 1_000_000,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs_root", type=Path)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    result = {name: summarize(args.runs_root / name) for name in ("mb80r1", "mc80r2")}
    result["reduction"] = {
        "mean_ms": result["mb80r1"]["mean_ms"] - result["mc80r2"]["mean_ms"],
        "union_ms": result["mb80r1"]["union_ms"] - result["mc80r2"]["union_ms"],
    }
    args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
