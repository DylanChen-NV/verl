#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

RECOMPUTE_MARKER = "VERL_RECOMPUTE_EVENT "
STRICT_MARKER = "MOONCAKE_STRICT_REUSE_EVENT "
RETRY_SUFFIX = "__verl_recompute_attempt_"


def payload(line, marker):
    if marker not in line:
        return None
    try:
        return json.loads(line.split(marker, 1)[1])
    except json.JSONDecodeError:
        return None


def engine_id(request_id):
    return request_id.rsplit("-", 1)[0]


def logical_id(request_id):
    return request_id.split(RETRY_SUFFIX, 1)[0]


def server_ip(server_id):
    return server_id.rsplit(":", 1)[0]


def key_digest(value):
    # Ignore the transfer-batch index; it can change when loads are combined.
    return value.split(":", 1)[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path, required=True)
    args = parser.parse_args()
    rpc_servers = defaultdict(list)
    first_output = set()
    strict_events = defaultdict(list)
    sends_by_digest = defaultdict(list)
    phase_counts = defaultdict(int)
    transfer_failures = 0
    for path in args.logs:
        for line in path.read_text(errors="replace").splitlines():
            transfer_failures += int("TRANSFER_FAIL" in line or "failed to copy from CUDA" in line)
            event = payload(line, RECOMPUTE_MARKER)
            if event:
                request_id = event.get("logical_request_id") or event.get("engine_request_id")
                if request_id and event.get("phase") == "SERVER_RPC_BEGIN":
                    rpc_servers[request_id].append(event["server_id"])
                if request_id and event.get("phase") == "FIRST_OUTPUT":
                    first_output.add(event.get("engine_request_id") or request_id)
            event = payload(line, STRICT_MARKER)
            if event:
                phase = event.get("phase", "unknown")
                phase_counts[phase] += 1
                request_id = event.get("request_id")
                if request_id:
                    strict_events[engine_id(request_id)].append(event)
                if phase == "FINISHED_SEND" and event.get("success") is True:
                    for digest in event.get("key_digests", []):
                        sends_by_digest[key_digest(digest)].append(event)

    rows = []
    for retry_id in sorted(key for key in rpc_servers if RETRY_SUFFIX in key):
        base_id = logical_id(retry_id)
        if not rpc_servers.get(base_id) or not rpc_servers.get(retry_id):
            continue
        original_ip = server_ip(rpc_servers[base_id][0])
        consumer_ip = server_ip(rpc_servers[retry_id][-1])
        receives = [event for event in strict_events.get(retry_id, []) if event.get("phase") == "FINISHED_RECV" and event.get("node_ip") == consumer_ip and event.get("success") is True]
        recv_digests = {key_digest(digest) for event in receives for digest in event.get("key_digests", [])}
        sources = {digest: [event for event in sends_by_digest[digest] if event.get("node_ip") != consumer_ip] for digest in recv_digests}
        shared = {digest for digest, events in sources.items() if events}
        producer_ips = sorted({event["node_ip"] for digest in shared for event in sources[digest]})
        rows.append({
            "logical_request_id": base_id,
            "retry_engine_request_id": retry_id,
            "original_request_ip": original_ip,
            "consumer_ip": consumer_ip,
            "retry_cross_node": original_ip != consumer_ip,
            "digest_producer_ips": ",".join(producer_ips),
            "producer_send_events": sum(len(sources[digest]) for digest in shared),
            "consumer_recv_events": len(receives),
            "shared_digest_count": len(shared),
            "recv_hit_tokens": max((int(event.get("kvpool_cached_tokens", 0)) for event in receives), default=0),
            "first_output": retry_id in first_output,
            "strict_cross_node_reuse": bool(original_ip != consumer_ip and receives and shared and retry_id in first_output),
        })
    proven = [row for row in rows if row["strict_cross_node_reuse"]]
    summary = {
        "schema_version": 1,
        "input_logs": [str(path) for path in args.logs],
        "transfer_failures": transfer_failures,
        "strict_phase_counts": dict(sorted(phase_counts.items())),
        "retry_request_count": len(rows),
        "retry_recv_request_count": sum(row["consumer_recv_events"] > 0 for row in rows),
        "strict_cross_node_reuse_request_count": len(proven),
        "strict_cross_node_reuse": bool(proven) and transfer_failures == 0,
        "requests": rows,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with args.csv_out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({key: value for key, value in summary.items() if key != "requests"}, indent=2))


if __name__ == "__main__":
    main()
