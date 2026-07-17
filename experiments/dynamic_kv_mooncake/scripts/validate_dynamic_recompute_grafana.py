#!/usr/bin/env python3
"""Validate a static dynamic-recompute Grafana export without Grafana."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_rows(target):
    return list(csv.DictReader(io.StringIO(target["csvContent"])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    acceptance = json.loads((output / "grafana_export_acceptance.json").read_text())
    dashboard_path = output / "verl_dynamic_recompute_dashboard.json"
    dashboard = json.loads(dashboard_path.read_text())
    assert acceptance["accepted"]
    assert sha256(dashboard_path) == acceptance["dashboard_sha256"]
    assert dashboard["uid"] == acceptance["dashboard_uid"]
    assert dashboard["graphTooltip"] == 1
    assert len(dashboard["panels"]) == acceptance["panel_count"] == 6
    panels = {panel["title"]: panel for panel in dashboard["panels"]}
    trajectory = panels["Retry trajectories by serving server"]
    assert len(trajectory["targets"]) == acceptance["trajectory_track_count"]
    assert all(csv_rows(item) for item in trajectory["targets"])
    events = panels["Dynamic cycle and retry events"]
    assert len(events["targets"]) == acceptance["event_phase_count"]
    assert all(csv_rows(item) for item in events["targets"])
    active = csv_rows(panels["Active generate attempts by serving server"]["targets"][0])
    assert active and len(active[0]) == acceptance["serving_server_count"] + 1
    attempts = csv_rows(panels["Retry attempt attribution"]["targets"][0])
    assert len(attempts) == acceptance["retry_attempt_count"]
    assert sum(row["status"] == "completed" for row in attempts) == acceptance["completed_retry_count"]
    assert sum(row["status"] == "reaborted" for row in attempts) == acceptance["reaborted_retry_count"]
    assert all(row["strict_contained"] == "True" for row in attempts if row["status"] == "completed")
    result = {
        "accepted": True, "dashboard_uid": dashboard["uid"],
        "dashboard_sha256": sha256(dashboard_path), "panel_count": len(dashboard["panels"]),
        "trajectory_track_count": len(trajectory["targets"]), "retry_attempt_count": len(attempts),
        "shared_crosshair": True, "all_targets_have_rows": True,
    }
    (output / "grafana_static_validation.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
