#!/usr/bin/env python3
"""Export the accepted recompute trajectory data as a static Grafana dashboard."""

import argparse
import csv
import datetime
import hashlib
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


DATASOURCE_TYPE = "grafana-testdata-datasource"
DATASOURCE_UID = "verl-dynamic-recompute-static"
EVENT_PREFIX = "VERL_RECOMPUTE_EVENT "


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iso_time_ns(epoch_ns):
    value = datetime.datetime.fromtimestamp(epoch_ns / 1e9, datetime.timezone.utc)
    return value.strftime("%Y-%m-%dT%H:%M:%S.") + "%06dZ" % value.microsecond


def csv_text(header, rows):
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def target(ref_id, content, alias=None):
    value = {
        "refId": ref_id,
        "scenarioId": "csv_content",
        "csvContent": content,
        "datasource": {"type": DATASOURCE_TYPE, "uid": DATASOURCE_UID},
    }
    if alias:
        value["alias"] = alias
    return value


def parse_events(path):
    events = []
    with path.open(errors="replace") as handle:
        for line in handle:
            marker = line.find(EVENT_PREFIX)
            if marker < 0:
                continue
            try:
                event = json.loads(line[marker + len(EVENT_PREFIX):].strip())
            except (TypeError, ValueError):
                continue
            if isinstance(event, dict) and event.get("wall_ns") is not None:
                events.append(event)
    return events


def load_joined(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_spans(path, joined):
    csv.field_size_limit(sys.maxsize)
    observed = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["name"] != "LLMServerClient.generate":
                continue
            attrs = json.loads(row["attributes_json"])
            logical_id = attrs.get("verl.recompute.logical_request_id")
            attempt_id = attrs.get("verl.recompute.attempt_id")
            if logical_id is None or attempt_id is None:
                continue
            key = (str(logical_id), int(attempt_id))
            if key in observed:
                raise AssertionError("duplicate generate span: %r" % (key,))
            observed[key] = (row["span_id"], attrs.get("verl.rollout.server_id"))
    expected = {
        (row["logical_request_id"], int(row["attempt_id"])):
        (row["span_id"], row["server_id"])
        for row in joined
    }
    if observed != expected:
        raise AssertionError("spans.csv and joined attempts differ")
    return len(observed)


def timeline_points(intervals):
    points = []
    for index, (start_ns, end_ns, state) in enumerate(sorted(intervals)):
        points.append([iso_time_ns(start_ns), state])
        next_start = sorted(intervals)[index + 1][0] if index + 1 < len(intervals) else None
        if next_start is None or next_start > end_ns:
            points.append([iso_time_ns(end_ns), ""])
    return points


def active_rows(intervals_by_server, servers, start_ns, end_ns):
    deltas = defaultdict(lambda: defaultdict(int))
    for server, intervals in intervals_by_server.items():
        for begin, end in intervals:
            deltas[begin][server] += 1
            deltas[end][server] -= 1
    deltas[start_ns]
    deltas[end_ns]
    active = dict((server, 0) for server in servers)
    rows = []
    for timestamp in sorted(deltas):
        for server, delta in deltas[timestamp].items():
            active[server] += delta
        if min(active.values()) < 0:
            raise AssertionError("negative active generate counter")
        rows.append([iso_time_ns(timestamp)] + [active[server] for server in servers])
    if any(active.values()):
        raise AssertionError("active generate counters do not return to zero")
    return rows


def state_panel(panel_id, title, description, grid, targets):
    return {
        "id": panel_id, "type": "state-timeline", "title": title,
        "description": description, "gridPos": grid,
        "datasource": {"type": DATASOURCE_TYPE, "uid": DATASOURCE_UID},
        "targets": targets,
        "fieldConfig": {"defaults": {"custom": {"fillOpacity": 85, "lineWidth": 0}}, "overrides": []},
        "options": {"alignValue": "left", "mergeValues": True, "rowHeight": 0.45,
                    "showValue": "auto", "tooltip": {"mode": "single", "sort": "none"},
                    "legend": {"displayMode": "list", "placement": "bottom", "showLegend": True}},
    }


def table_panel(panel_id, title, description, grid, content):
    return {
        "id": panel_id, "type": "table", "title": title, "description": description,
        "gridPos": grid, "datasource": {"type": DATASOURCE_TYPE, "uid": DATASOURCE_UID},
        "targets": [target("A", content)], "fieldConfig": {"defaults": {}, "overrides": []},
        "options": {"showHeader": True, "cellHeight": "sm", "footer": {"show": False}},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spans-csv", required=True, type=Path)
    parser.add_argument("--joined-csv", required=True, type=Path)
    parser.add_argument("--joined-summary", required=True, type=Path)
    parser.add_argument("--strict-report", required=True, type=Path)
    parser.add_argument("--raw-log", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    joined = load_joined(args.joined_csv)
    strict = json.loads(args.strict_report.read_text())
    joined_summary = json.loads(args.joined_summary.read_text())
    events = parse_events(args.raw_log)
    span_count = validate_spans(args.spans_csv, joined)

    by_logical = defaultdict(list)
    intervals_by_server = defaultdict(list)
    for row in joined:
        row["attempt_id_int"] = int(row["attempt_id"])
        row["span_start_ns_int"] = int(row["span_start_ns"])
        row["span_end_ns_int"] = int(row["span_end_ns"])
        by_logical[row["logical_request_id"]].append(row)
        intervals_by_server[row["server_id"]].append(
            (row["span_start_ns_int"], row["span_end_ns_int"])
        )

    retry_rows = sorted(
        (row for row in joined if row["attempt_id_int"] > 0),
        key=lambda row: (row["scheduled_wall_ns"] or row["span_start_ns"], row["logical_request_id"]),
    )
    if len(retry_rows) != strict["attempt_count"]:
        raise AssertionError("retry count differs from strict report")

    trajectory_targets = []
    trajectory_audit = []
    retry_logical_ids = sorted(
        {row["logical_request_id"] for row in retry_rows},
        key=lambda logical_id: min(row["span_start_ns_int"] for row in by_logical[logical_id] if row["attempt_id_int"] > 0),
    )
    for index, logical_id in enumerate(retry_logical_ids):
        rows = sorted(by_logical[logical_id], key=lambda row: row["attempt_id_int"])
        attempt_ids = [row["attempt_id_int"] for row in rows]
        if attempt_ids != list(range(attempt_ids[-1] + 1)):
            raise AssertionError("retry trajectory has a missing attempt: %s" % logical_id)
        intervals = []
        for row in rows:
            state = "attempt%d %s @ %s" % (row["attempt_id_int"], row["status"], row["server_id"])
            intervals.append((row["span_start_ns_int"], row["span_end_ns_int"], state))
            trajectory_audit.append([
                logical_id, row["attempt_id"], row["status"], row["server_id"],
                iso_time_ns(row["span_start_ns_int"]), iso_time_ns(row["span_end_ns_int"]),
                row["partial_tokens"], row["retry_prefix_tokens"], row["put_status"],
            ])
        alias = "%03d / %s" % (index, logical_id[:12])
        trajectory_targets.append(target("T%03d" % index, csv_text(["time", alias], timeline_points(intervals)), alias))

    event_phases = ["CYCLE_START", "RETRY_SUBMIT", "SCHEDULED", "FIRST_OUTPUT", "CLIENT_REABORT", "CYCLE_END"]
    event_targets = []
    event_audit = []
    for index, phase in enumerate(event_phases):
        phase_events = sorted((event for event in events if event.get("phase") == phase), key=lambda event: event["wall_ns"])
        if not phase_events:
            continue
        points = []
        for event in phase_events:
            timestamp = int(event["wall_ns"])
            logical = str(event.get("logical_request_id", ""))[:8]
            value = phase if not logical else "%s %s" % (phase, logical)
            points.extend([[iso_time_ns(timestamp), value], [iso_time_ns(timestamp + 1000000), ""]])
            event_audit.append([phase, iso_time_ns(timestamp), event.get("logical_request_id", ""), event.get("attempt_id", ""), event.get("node_id", "")])
        event_targets.append(target("E%02d" % index, csv_text(["time", phase], points), phase))

    servers = sorted(intervals_by_server)
    window_start_ns = min(row["span_start_ns_int"] for row in joined)
    window_end_ns = max(row["span_end_ns_int"] for row in joined)
    active = active_rows(intervals_by_server, servers, window_start_ns, window_end_ns)
    active_csv = csv_text(["time"] + servers, active)

    status_counts = Counter(row["status"] for row in retry_rows)
    server_summary_rows = []
    for server in servers:
        server_rows = [row for row in joined if row["server_id"] == server]
        retry_server_rows = [row for row in server_rows if row["attempt_id_int"] > 0]
        server_summary_rows.append([
            server, server.rsplit(":", 1)[0], len(server_rows), len(retry_server_rows),
            sum(row["status"] == "completed" for row in retry_server_rows),
            sum(row["status"] == "reaborted" for row in retry_server_rows),
            max(row[servers.index(server) + 1] for row in active),
        ])
    server_summary_csv = csv_text(
        ["server_id", "node", "attempts", "retries", "completed", "reaborted", "peak_active"],
        server_summary_rows,
    )

    attempt_rows = [[
        row["logical_request_id"], row["attempt_id"], row["dynamic_cycle_id"], row["status"],
        row["initial_server_id"], row["server_id"], row["cross_node_from_initial"],
        row["partial_tokens"], row["retry_prefix_tokens"], row["policy_version"], row["put_status"],
        row["queue_time_ms"], row["queue_excluded_time_ms"], row["strict_interval_contained"],
    ] for row in retry_rows]
    attempts_csv = csv_text(
        ["logical_request_id", "attempt_id", "cycle", "status", "initial_server", "retry_server",
         "cross_node", "partial_tokens", "retry_prefix_tokens", "policy_version", "put_status",
         "queue_ms", "recompute_ms", "strict_contained"], attempt_rows,
    )
    stats_csv = csv_text(["metric", "value"], [
        ["all generate attempts", len(joined)], ["retry attempts", len(retry_rows)],
        ["completed retries", status_counts["completed"]], ["reaborted retries", status_counts["reaborted"]],
        ["serving servers", len(servers)], ["cross-node retries", joined_summary["cross_node_retry_count"]],
        ["strict recompute union ms", strict["cycles"][0]["nodes"][0]["queue_excluded"]["union_ms"]],
        ["strict recompute mean ms", strict["distributions"]["queue_excluded_time_ms"]["mean"]],
    ])

    panels = [
        state_panel(1, "Retry trajectories by serving server", "Each row is one logical request with initial and retry attempts on the observed serving server.", {"x": 0, "y": 0, "w": 24, "h": 18}, trajectory_targets),
        state_panel(2, "Dynamic cycle and retry events", "Absolute wall-clock events from the strict recompute instrumentation.", {"x": 0, "y": 18, "w": 24, "h": 9}, event_targets),
        {"id": 3, "type": "timeseries", "title": "Active generate attempts by serving server", "description": "Concurrency reconstructed from MLflow generate span intervals.", "gridPos": {"x": 0, "y": 27, "w": 24, "h": 9}, "datasource": {"type": DATASOURCE_TYPE, "uid": DATASOURCE_UID}, "targets": [target("A", active_csv)], "fieldConfig": {"defaults": {"min": 0, "custom": {"drawStyle": "line", "lineInterpolation": "stepAfter", "lineWidth": 2, "fillOpacity": 7, "showPoints": "never"}}, "overrides": []}, "options": {"legend": {"displayMode": "table", "placement": "right", "calcs": ["max", "mean"]}, "tooltip": {"mode": "multi", "sort": "desc"}}},
        table_panel(4, "Retry attempt attribution", "Strict retry rows joined to MLflow spans and serving server IDs.", {"x": 0, "y": 36, "w": 24, "h": 12}, attempts_csv),
        table_panel(5, "Serving server summary", "Generate and retry load by observed vLLM server.", {"x": 0, "y": 48, "w": 15, "h": 8}, server_summary_csv),
        table_panel(6, "Acceptance summary", "Counts and strict queue-excluded recompute measurements.", {"x": 15, "y": 48, "w": 9, "h": 8}, stats_csv),
    ]
    uid = "verl-dyn-recompute-%s" % args.run_id[-8:]
    dashboard = {
        "id": None, "uid": uid, "title": "VeRL dynamic recompute trajectory",
        "description": "Static A/B/C trajectory view generated directly from MLflow spans and strict recompute data.",
        "tags": ["verl", "dynamic-resource", "recompute", "trajectory"], "timezone": "utc",
        "schemaVersion": 41, "version": 1, "refresh": False, "graphTooltip": 1,
        "time": {"from": iso_time_ns(window_start_ns - 1000000000), "to": iso_time_ns(window_end_ns + 1000000000)},
        "timepicker": {"hidden": False}, "templating": {"list": []}, "annotations": {"list": []},
        "links": [], "panels": panels,
    }
    dashboard_path = output / "verl_dynamic_recompute_dashboard.json"
    write_json(dashboard_path, dashboard)
    (output / "active_generate_by_server.csv").write_text(active_csv)
    (output / "retry_attempts.csv").write_text(attempts_csv)
    (output / "server_summary.csv").write_text(server_summary_csv)
    (output / "acceptance_summary.csv").write_text(stats_csv)
    with (output / "trajectory_timeline.csv").open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["logical_request_id", "attempt_id", "status", "server_id", "start_time", "end_time", "partial_tokens", "retry_prefix_tokens", "put_status"]); writer.writerows(trajectory_audit)
    with (output / "dynamic_events.csv").open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["phase", "time", "logical_request_id", "attempt_id", "node_id"]); writer.writerows(event_audit)

    provisioning = output / "provisioning"
    (provisioning / "datasources").mkdir(parents=True, exist_ok=True)
    (provisioning / "dashboards").mkdir(parents=True, exist_ok=True)
    (provisioning / "datasources" / "dynamic_static.yaml").write_text(
        "apiVersion: 1\ndatasources:\n  - name: VeRL dynamic recompute static data\n    type: %s\n    uid: %s\n    access: proxy\n    isDefault: true\n" % (DATASOURCE_TYPE, DATASOURCE_UID))
    (provisioning / "dashboards" / "dynamic_static.yaml").write_text(
        "apiVersion: 1\nproviders:\n  - name: VeRL dynamic recompute\n    folder: VeRL dynamic recompute\n    type: file\n    disableDeletion: false\n    allowUiUpdates: true\n    updateIntervalSeconds: 10\n    options:\n      path: %s\n" % output)

    accepted = (
        span_count == len(joined)
        and len(retry_rows) == strict["attempt_count"]
        and status_counts == Counter(strict["status_counts"])
        and len(trajectory_targets) > 0 and len(servers) == 8
        and all(row["strict_interval_contained"] == "True" for row in retry_rows if row["status"] == "completed")
    )
    acceptance = {
        "accepted": accepted, "run_id": args.run_id, "dashboard_uid": uid,
        "dashboard_sha256": sha256(dashboard_path), "joined_sha256": sha256(args.joined_csv),
        "spans_sha256": sha256(args.spans_csv), "strict_report_sha256": sha256(args.strict_report),
        "generate_attempt_count": len(joined), "retry_attempt_count": len(retry_rows),
        "completed_retry_count": status_counts["completed"], "reaborted_retry_count": status_counts["reaborted"],
        "serving_server_count": len(servers), "trajectory_track_count": len(trajectory_targets),
        "event_phase_count": len(event_targets), "panel_count": len(panels),
    }
    write_json(output / "grafana_export_acceptance.json", acceptance)
    if not accepted:
        raise AssertionError(acceptance)
    print(json.dumps(acceptance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
