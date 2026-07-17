#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


EVENT_PREFIX = "VERL_RECOMPUTE_EVENT "


def parse_events(path):
    events = []
    with path.open(errors="replace") as handle:
        for line in handle:
            marker = line.find(EVENT_PREFIX)
            if marker < 0:
                continue
            try:
                event = json.loads(line[marker + len(EVENT_PREFIX) :].strip())
            except (TypeError, ValueError):
                continue
            if isinstance(event, dict):
                events.append(event)
    return events


def attempt_key(logical_request_id, attempt_id):
    return (str(logical_request_id), int(attempt_id))


def server_node(server_id):
    return str(server_id).rsplit(":", 1)[0] if server_id else ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spans-csv", required=True, type=Path)
    parser.add_argument("--strict-report", required=True, type=Path)
    parser.add_argument("--raw-log", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    strict_report = json.loads(args.strict_report.read_text())
    strict_by_key = {}
    for attempt in strict_report["attempts"]:
        key = attempt_key(attempt["logical_request_id"], attempt["attempt_id"])
        if key in strict_by_key:
            raise AssertionError("duplicate strict attempt: {}".format(key))
        strict_by_key[key] = attempt

    events_by_key = defaultdict(lambda: defaultdict(list))
    for event in parse_events(args.raw_log):
        logical_id = event.get("logical_request_id")
        attempt_id = event.get("attempt_id")
        if logical_id is None or attempt_id is None:
            continue
        events_by_key[attempt_key(logical_id, attempt_id)][str(event.get("phase"))].append(event)

    csv.field_size_limit(sys.maxsize)
    span_by_key = {}
    span_rows = []
    with args.spans_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["name"] != "LLMServerClient.generate":
                continue
            attributes = json.loads(row["attributes_json"])
            logical_id = attributes.get("verl.recompute.logical_request_id")
            attempt_id = attributes.get("verl.recompute.attempt_id")
            if logical_id is None or attempt_id is None:
                continue
            key = attempt_key(logical_id, attempt_id)
            if key in span_by_key:
                raise AssertionError("duplicate MLflow attempt span: {}".format(key))
            span = {
                "trace_id": row["trace_id"],
                "span_id": row["span_id"],
                "span_start_ns": int(row["start_time_ns"]),
                "span_end_ns": int(row["end_time_ns"]),
                "attributes": attributes,
            }
            span_by_key[key] = span
            span_rows.append((key, span))

    initial_by_logical = {}
    for key, span in span_rows:
        if key[1] == 0:
            initial_by_logical[key[0]] = span

    joined = []
    containment_failures = []
    missing_strict_spans = []
    cross_node_retries = 0
    server_transitions = Counter()
    status_counts = Counter()
    for key, span in sorted(span_rows):
        logical_id, attempt_id = key
        strict = strict_by_key.get(key)
        status = "initial" if attempt_id == 0 else (strict or {}).get("status", "unmatched_retry")
        status_counts[status] += 1
        attributes = span["attributes"]
        server_id = str(attributes.get("verl.rollout.server_id", ""))
        initial_span = initial_by_logical.get(logical_id)
        initial_server_id = "" if initial_span is None else str(
            initial_span["attributes"].get("verl.rollout.server_id", "")
        )
        if attempt_id > 0:
            server_transitions["{} -> {}".format(initial_server_id, server_id)] += 1
        cross_node = bool(
            attempt_id > 0
            and initial_server_id
            and server_id
            and server_node(initial_server_id) != server_node(server_id)
        )
        if cross_node:
            cross_node_retries += 1

        phases = events_by_key.get(key, {})
        submit = phases.get("SUBMIT", [])
        scheduled = phases.get("SCHEDULED", [])
        first_output = phases.get("FIRST_OUTPUT", [])
        reabort = phases.get("CLIENT_REABORT", [])
        submit_wall_ns = submit[0].get("wall_ns") if len(submit) == 1 else None
        scheduled_wall_ns = scheduled[0].get("wall_ns") if len(scheduled) == 1 else None
        first_output_wall_ns = first_output[0].get("wall_ns") if len(first_output) == 1 else None
        reabort_wall_ns = reabort[0].get("wall_ns") if len(reabort) == 1 else None

        contained = None
        if status == "completed":
            contained = bool(
                scheduled_wall_ns is not None
                and first_output_wall_ns is not None
                and span["span_start_ns"] <= int(scheduled_wall_ns) < int(first_output_wall_ns) <= span["span_end_ns"]
            )
            if not contained:
                containment_failures.append(key)

        row = {
            "logical_request_id": logical_id,
            "attempt_id": attempt_id,
            "dynamic_cycle_id": attributes.get("verl.recompute.dynamic_cycle_id"),
            "status": status,
            "trace_id": span["trace_id"],
            "span_id": span["span_id"],
            "span_start_ns": span["span_start_ns"],
            "span_end_ns": span["span_end_ns"],
            "submit_wall_ns": submit_wall_ns,
            "scheduled_wall_ns": scheduled_wall_ns,
            "first_output_wall_ns": first_output_wall_ns,
            "reabort_wall_ns": reabort_wall_ns,
            "strict_interval_contained": contained,
            "server_id": server_id,
            "server_node": server_node(server_id),
            "initial_server_id": initial_server_id,
            "initial_server_node": server_node(initial_server_id),
            "cross_node_from_initial": cross_node,
            "engine_request_id": attributes.get("verl.recompute.engine_request_id"),
            "partial_tokens": attributes.get("verl.recompute.partial_tokens"),
            "retry_prefix_tokens": attributes.get("verl.recompute.retry_prefix_tokens"),
            "policy_version": attributes.get("verl.recompute.policy_version"),
            "put_status": attributes.get("verl.flexkv.put_status"),
            "queue_time_ms": None if strict is None else strict.get("queue_time_ms"),
            "queue_excluded_time_ms": None if strict is None else strict.get("queue_excluded_time_ms"),
        }
        joined.append(row)

    for key in strict_by_key:
        if key not in span_by_key:
            missing_strict_spans.append(key)

    summary = {
        "mlflow_attempt_count": len(span_by_key),
        "initial_attempt_count": status_counts["initial"],
        "strict_attempt_count": len(strict_by_key),
        "matched_strict_attempt_count": len(strict_by_key) - len(missing_strict_spans),
        "status_counts": dict(sorted(status_counts.items())),
        "completed_containment_count": sum(
            1 for row in joined if row["status"] == "completed" and row["strict_interval_contained"]
        ),
        "containment_failures": containment_failures,
        "missing_strict_spans": missing_strict_spans,
        "server_ids": dict(sorted(Counter(row["server_id"] for row in joined).items())),
        "cross_node_retry_count": cross_node_retries,
        "server_transition_counts": dict(sorted(server_transitions.items())),
    }

    assert len(span_by_key) == status_counts["initial"] + len(strict_by_key), summary
    assert status_counts["initial"] == 160, summary
    assert not missing_strict_spans, summary
    assert not containment_failures, summary
    expected_status_counts = Counter(attempt["status"] for attempt in strict_by_key.values())
    for status, count in expected_status_counts.items():
        assert status_counts[status] == count, summary

    csv_path = args.output_dir / "trajectory_recompute_attempts.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(joined[0].keys()))
        writer.writeheader()
        writer.writerows(joined)
    summary_path = args.output_dir / "trajectory_recompute_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
