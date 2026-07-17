# Mooncake Dynamic Resource A/B/C Results (2026-07-17)

## Conclusion

Mooncake abort KV offload and cross-node reuse works in the sleep-enabled
dynamic hybrid flow. In formal C run `mc80r2`, all 26 target retries were
strictly proven to reuse KV produced on `10.65.4.75` from consumers on
`10.65.4.69`. The proof joins original SEND and retry RECV key digests, server
provenance, and successful FIRST_OUTPUT. There were no transfer failures.

The current implementation is functionally correct but not performance-ready.
Saving 26 approximately 8K prefixes took about 30 seconds at the barrier. This
dominates the 275 ms target-cohort queue-excluded recompute-union reduction.

## Fixed Environment

```text
cluster: DFW
resource: 2 nodes x 8 H100
persistent session: dyn-moon-final-dispatch-20260717
Slurm job: 14096137
nodes: pool0-00340, pool0-00341
image: verlai/verl:vllm023.dev1
model: Qwen3-30B-A3B, BF16
trainer: Megatron, TP2/PP2/EP4
rollout: vLLM 0.23 development image, TP2
dataset: DAPO-Math-17K
response length: 8192
rollout.total_rollout_steps: 80
concurrent_samples_per_replica: 32
rollout.max_num_seqs: 32
dynamic_scaling_deactivate_ratio: 0.0625
sleep/free cache: enabled
```

All formal groups used the same isolated VERL/vLLM/Mooncake code and container.
Only the KV mode changed:

```text
A: abort KV reuse disabled
B: Mooncake PUT/barrier enabled, consumer lookup forced miss
C: Mooncake PUT/barrier and real lookup/load enabled
```

## Runs

| Group | Run ID | Persistent command | Result |
|---|---|---|---|
| A | `ma80r1` | `cmd-20260717-093957-depo` | 160 traces, 101 retries, pass |
| B | `mb80r1` | `cmd-20260717-095904-tnb4` | 160 traces, 89 retries, pass |
| C diagnostic | `mc80r1` | `cmd-20260717-101921-b6lz` | pass, but dynamic barrier had zero target requests; excluded |
| C formal | `mc80r2` | `cmd-20260717-103928-2mz4` | 160 traces, 90 retries, 26 strict cross-node hits, pass |

`mc80r1` is intentionally retained. Its four `BARRIER_VISIBLE` events all had
`request_count=0`, and its RECV events were only 48-token startup prompt hits.
It must not be used as formal reuse evidence.

## Strict C Evidence

```text
dynamic barrier request counts: 5 + 5 + 16 + 0 = 26
FINAL_SAVE_QUEUED: 26
FINAL_SAVE_DISPATCHED: 26
strict FINISHED_RECV events: 54 (two TP ranks plus startup events)
strict cross-node reuse requests: 26
producer IP: 10.65.4.75
consumer IP: 10.65.4.69
transfer failures: 0
```

For the 26 proven requests:

| Metric | Result |
|---|---:|
| Reused tokens, mean | 8236.31 |
| Reused tokens, min/max | 8096 / 8336 |
| Hit ratio, mean | 99.895% |
| Hit ratio, minimum | 99.804% |
| Newly scheduled tokens, mean | 8.65 |
| Newly scheduled tokens, min/max | 1 / 16 |
| Queue-excluded time, mean | 26.31 ms |
| Queue-excluded union | 551.96 ms |
| Queue time, mean | 10.282 s |

This proves reuse of the large aborted partial prefix, rather than a short
sibling-prompt cache hit.

## Recompute Comparison

Whole-run first-cycle report:

| Group | Completed | Re-aborted | Queue-excluded union |
|---|---:|---:|---:|
| A | 69 | 32 | 2195.02 ms |
| B | 57 | 32 | 1640.33 ms |
| C | 58 | 32 | 1381.46 ms |

Whole-run unions are descriptive only because the independent runs contain
different target counts. The cross-node target-cohort comparison is the useful
B/C result:

| Group | Target requests | Mean | Union |
|---|---:|---:|---:|
| B forced miss | 25 | 251.32 ms | 827.16 ms |
| C real hit | 26 | 26.31 ms | 551.96 ms |
| Reduction | - | 225.01 ms (89.5%) | 275.20 ms (33.3%) |

Queue-excluded means `[T_scheduled, T_first_output]`; queue/load time before
scheduling is deliberately excluded. Re-aborted attempts are not included.

## Offload Cost And Net Result

```text
A dynamic cycle: 0.510 s
B dynamic cycle: 29.133 s
C dynamic cycle: 30.393 s
C max per-replica barrier wait: 29.969 s
```

The 16-request replica determines the C barrier wall, while two other replicas
held five requests each and one held none. For Qwen3-30B-A3B BF16, theoretical
KV bytes per token are:

```text
bytes_per_token = layers * 2(K,V) * kv_heads * head_dim * bytes_per_value
                = 48 * 2 * 4 * 128 * 2
                = 98,304 bytes
```

Using 26 requests and mean 8236 reused tokens gives about 21.05 GB of KV. The
approximately 30.09 second PUT-visible wall corresponds to only about 0.70 GB/s
end-to-end effective throughput. This is an estimate based on model KV shape;
it is not a raw Mooncake link counter.

The design formula is:

```text
net_cycle_gain = recompute_saved - deactivate_increment - get_load_overhead
```

Using target-cohort union reduction and B versus A cycle increment:

```text
recompute_saved       = 0.275 s
deactivate_increment  = 29.133 - 0.510 = 28.623 s
best_case_net_gain    = 0.275 - 28.623 = -28.348 s
```

This best case already ignores get/load overhead, so the present end-to-end net
gain is definitively negative. C target queue time also averages 10.282 seconds;
it includes connector load plus serving queue and is not yet decomposed into a
pure GET metric.

## Reproduction Entry Points

```text
scripts/run_persistent_moon_a_formal_r80.sh
scripts/run_persistent_moon_b_formal_r80.sh
scripts/run_persistent_moon_c_formal_r80_r2.sh
scripts/postprocess_mooncake_formal_run.sh
scripts/analyze_mooncake_strict_reuse.py
scripts/summarize_mooncake_proven_reuse.py
scripts/compare_mooncake_target_cohorts.py
```

Remote evidence root:

```text
/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/05_claude_ws/dynamic_mooncake_abort_dev/monitoring/runs/{ma80r1,mb80r1,mc80r2}
```

Important summary artifacts:

```text
analysis/queue_excluded_report.json
analysis/formal_run_summary.json
analysis/strict_cross_node_reuse_report.json
analysis/strict_cross_node_reuse_requests.csv
analysis/proven_reuse_summary.json
monitoring/mooncake_abc_target_comparison.json
trace_dump/trace_acceptance_summary.json
```

## Grafana Acceptance

All three runs were joined against MLflow and exported as static Grafana dashboards. Each export passed offline validation with eight serving servers, six panels, shared crosshair, non-empty targets, no missing strict spans, and no completed-interval containment failures.

```text
A dashboard SHA256: e9f7d793e8aa82e16a26cb51e9f252ef75857bb8b627a8bc8485b4530d672f08
B dashboard SHA256: 3c24fb1ddd5a381d780c0d87e7c65a53e62dc99b3f872003da1621d892901e07
C dashboard SHA256: ab7aab9986f9743f2ef16c9bf587c4860d7ed008fc0456cab99780a1b3e1cade
```

C was also provisioned in Grafana 13.0.0 and passed HTTP service validation: all targets in all six panels returned non-empty frames with no query errors.

```text
dashboard UID: verl-dyn-recompute-mc80r2
remote endpoint: http://127.0.0.1:31598
service evidence: mc80r2/grafana/grafana_service_acceptance.json
```

No browser screenshot was captured; static and HTTP datasource acceptance are complete.

## Next Work

1. Make final PUT dispatch concurrent or bounded-parallel across requests and
   rebalance requests across hybrid replicas; the current 16-request replica
   serializes the barrier.
2. Add explicit PUT bytes/time and GET bytes/time metrics so transfer cost is
   separated from serving queue time.
3. Re-run matched B/C after transfer optimization; use target-cohort counts and
   prefix-token distributions as acceptance gates.
4. Optionally add a browser screenshot to the completed static and HTTP Grafana acceptance evidence.
5. Keep FlexKV support as a separate TODO until its vLLM sleep-compatible GPU
   registration/lifecycle is available; do not reuse the old conflicting patch.

Do not release the persistent DFW allocation without explicit user approval.
