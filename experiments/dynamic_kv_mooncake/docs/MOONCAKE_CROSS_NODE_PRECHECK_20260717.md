# Mooncake sleep-enabled dynamic hybrid cross-node precheck

Date: 2026-07-17 UTC

## Result

`mcpre1` proves strict cross-node KV reuse for an aborted dynamic hybrid request:

- original request ran on `pool0-00341` / `10.65.4.75`;
- abort final-save exported 8,288 complete-prefix tokens;
- retry ran on `pool0-00340` / `10.65.4.69`;
- retry received all 8,288 tokens and its first scheduler step computed only one new token;
- producer/consumer TP2 key digests match, no transfer failure occurred;
- sleep-enabled trainer step completed normally after the delayed-free barrier.

This is a functional precheck, not the A/B/C performance conclusion.

## Resource and runtime

```text
cluster: DFW
persistent session: dyn-moon-final-dispatch-20260717
Slurm job: 14096137
nodes: pool0-00340,pool0-00341
GPUs: 2 nodes x 8 H100
image: verlai/verl:vllm023.dev1
command: cmd-20260717-085316-f1ey
run id: mcpre1
group: C (Mooncake enabled, lookup enabled)
model: Qwen3-30B-A3B BF16
trainer: Megatron TP2/PP2/EP4/ETP1/CP1
rollout: vLLM TP2, sleep enabled
planned trainer steps: 1
rollout.total_rollout_steps: 16
expected MLflow traces: 32
n responses per prompt: 2
concurrent samples per replica: 32
rollout max num seqs: 32
max prompt length: 1024
max response length: 8192
dynamic deactivate ratio: 0.0625
deactivate grace: 0 s
GPU memory utilization: 0.70
```

The ratio was reduced from `0.25` because the earlier `mcfd1` diagnostic let the
hybrid request finish before deactivation. `0.0625` reliably targets an in-flight
hybrid request and must be identical across formal A/B/C.

## Strict evidence

Logical request:

```text
logical_request_id: 43cb7f7dde0f45e695efa24dd7871fff
original engine request: 43cb7f7dde0f45e695efa24dd7871fff-8c1741d6
original node: 10.65.4.75
retry node: 10.65.4.69
saved/received prefix tokens: 8288
shared digest count: 1032 (516 keys per TP rank)
retry cross-node: true
first output observed: true
transfer failures: 0
strict_cross_node_reuse: true
```

Ordering in wall-clock nanoseconds:

```text
CYCLE_START             1784279142272061708
BARRIER_WAIT            1784279142313104172
FINAL_SAVE_QUEUED       1784279142351501958
FINAL_SAVE_DISPATCHED   1784279142351937624
BARRIER_VISIBLE         1784279144678591947
retry LOOKUP            1784279145714661325
```

The original producer logged TP2 `FINISHED_SEND` for 8,288 tokens before the
barrier became visible. After barrier release, the other node logged TP2
`FINISHED_RECV` for the same 8,288-token key set. vLLM then reported
`initial_computed_tokens=8288` and `num_scheduled_tokens=1` for the retry.

## Acceptance artifacts

Remote root:

```text
/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/05_claude_ws/dynamic_mooncake_abort_dev
```

Key artifacts:

```text
monitoring/runs/mcpre1/mlflow.db
monitoring/runs/mcpre1/trace_dump/traces.jsonl
monitoring/runs/mcpre1/trace_dump/spans.csv
monitoring/runs/mcpre1/trace_dump/trace_acceptance_summary.json
monitoring/runs/mcpre1/trace_dump/dump_manifest.json
monitoring/runs/mcpre1/strict/report.json
monitoring/runs/mcpre1/strict/requests.csv
```

MLflow redump acceptance:

```text
trace_count: 32
outer_generate_count: 32
single_turn_count: 32
inner_attempt_count: 33
retry_attempt_count: 1
server replica count: 8
invalid attempts: 0
all spans have times: true
```

Rank-0 raw log:

```text
/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/05_claude_ws/.gpu-workflow/dfw/persist/dyn-moon-final-dispatch-20260717/logs/cmd-20260717-085316-f1ey.rank-0.out
```

The original command returned 1 only because post-processing hard-coded
`--minimum-retry-attempts 16`. The runner now exposes
`MINIMUM_RETRY_ATTEMPTS` with default 16; the one-request precheck sets it to 1.
Re-running only MLflow export (`cmd-20260717-091436-dn9w`) passed completely.

## Reproduction entrypoints

```text
scripts/run_persistent_moon_c_cross_node_precheck.sh
scripts/dfw_h100_dynamic_recompute_30b_r8k_req32_mooncake_abc.sh
scripts/analyze_mooncake_strict_reuse.py
scripts/redump_mcpre1_mlflow.sh
```

Do not release the persistent allocation during iterative A/B/C work without
explicit confirmation. Keep Mooncake changes isolated under
`next_tests/dynamic_recompute_trajectory_mooncake`; FlexKV remains a separate TODO.
