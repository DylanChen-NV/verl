# Rollout Perf Tool V0 Implementation Guide

This document records the first usable version of the rollout perf tool. It is
intended as a practical handoff note for future development and validation.

Related design documents:

- `docs/rollout_perf_tool_design_plan.md`
- `docs/rollout_perf_trace_schema_draft.md`

The V0 implementation covered here is the P0 collection and Perfetto
visualization path. Closed-loop replay is not implemented yet.

## Validated Baseline

Validated workload:

```text
workspace: /lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/05_claude_ws/exp_rollout_perf_trace_retool_workload_1k_osl
verl branch: rollout-perf-tool-dev
validated code through: 781ff40c Use global active counter sum semantics
image: verlai/verl:vllm020.dev1
cluster: DFW
nodes: 2
gpus_per_node: 8
recipe: recipe/retool
model: Qwen2.5-7B-Instruct
job_id: 12465958
job_state: COMPLETED
wandb: https://wandb.ai/czqing422-sjtu/verl-fully-async-smoke/runs/retool-vllm020-2n8g-12465958
```

Validated trace outputs:

```text
raw trace dir:
/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/05_claude_ws/exp_rollout_perf_trace_retool_workload_1k_osl/rollout_perf_trace/12465958

full Perfetto:
/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/05_claude_ws/exp_rollout_perf_trace_retool_workload_1k_osl/rollout_perf_trace/perfetto_12465958.json

current focused active-counter Perfetto:
/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/05_claude_ws/exp_rollout_perf_trace_retool_workload_1k_osl/rollout_perf_trace/perfetto_active_counters_global_window_args_12465958.json
```

Raw trace validation from the successful workload:

```text
rollout_perf_trace_records=5878
span/trajectory=96
span/llm_turn=490
span/tool_call=394
span/vllm_engine_request=490
```

## Implementation Layout

Core modules:

```text
verl/utils/rollout_perf/config.py
verl/utils/rollout_perf/context.py
verl/utils/rollout_perf/collector.py
verl/utils/rollout_perf/writer.py
verl/utils/rollout_perf/export_perfetto.py
verl/utils/rollout_perf/vllm_metrics.py
```

Integration points:

```text
verl/experimental/agent_loop/agent_loop.py
verl/experimental/agent_loop/tool_agent_loop.py
verl/workers/rollout/llm_server.py
verl/workers/rollout/vllm_rollout/vllm_async_server.py
```

The collector API is intentionally small:

- `init_rollout_perf(config, role)` creates a process-local JSONL writer.
- `push_trace_context(...)` propagates IDs such as trajectory, turn, and request.
- `start_span(name, payload)` writes one complete span record at finish time.
- `emit_counter(...)` respects request sampling.
- `emit_unsampled_counter(...)` bypasses request sampling and is used for engine
  internal counters that must represent the full engine workload.

The raw trace is the source of truth. Perfetto files are derived offline by
`export_perfetto.py`.

## Configuration

Main config object:

```yaml
actor_rollout_ref:
  rollout:
    perf_trace:
      enable: false
      output_dir: null
      format: jsonl
      capture_content: false
      max_samples_per_step_per_worker: null
      flush_interval_s: 5.0
      export_perfetto: false
      engine_internal:
        enable: false
        backend: auto
        sample_interval_ms: 500
```

Validated workload overrides:

```bash
actor_rollout_ref.rollout.perf_trace.enable=True
actor_rollout_ref.rollout.perf_trace.output_dir="${PERF_TRACE_DIR}"
actor_rollout_ref.rollout.perf_trace.max_samples_per_step_per_worker=null
actor_rollout_ref.rollout.perf_trace.engine_internal.enable=True
actor_rollout_ref.rollout.perf_trace.engine_internal.sample_interval_ms=100
+ray_kwargs.ray_init.runtime_env.env_vars.VERL_ROLLOUT_PERF_OUTPUT_DIR="${PERF_TRACE_DIR}"
```

Important behavior:

- `max_samples_per_step_per_worker=null` means all trajectories are traced.
- Request sampling is only for debugging. It should not be enabled for perf
  analysis because it samples the request dimension.
- vLLM engine counters are unsampled and represent the full local engine
  workload, even if request sampling is enabled.
- `capture_content=false` by default. Prompt text, tool args, and raw tool output
  are not captured unless this is explicitly changed.
- The tested scripts export Perfetto explicitly after training. The
  `export_perfetto` config field exists but should not be assumed to perform the
  validated export path automatically.

Environment variables used by the implementation:

```text
VERL_ROLLOUT_PERF_OUTPUT_DIR
VERL_ROLLOUT_PERF_RUN_ID
VERL_ROLLOUT_PERF_FLUSH_INTERVAL_S
VERL_ROLLOUT_PERF_ENGINE_INTERNAL_SAMPLE_INTERVAL_MS
```

`vllm_async_server.py` sets
`VERL_ROLLOUT_PERF_ENGINE_INTERNAL_SAMPLE_INTERVAL_MS` before vLLM engine
initialization so the vLLM engine-core subprocess can use the same interval.

## Running The Validated Workload

Entrypoint:

```text
/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/05_claude_ws/exp_rollout_perf_trace_retool_workload_1k_osl/scripts/run_workload_1k_osl.sh
```

Submit on DFW:

```bash
/home/scratch.ziqingc_gpu/06_codex_projects/01_template/gpu-workflow/bin/dfw-run submit \
  --nodes 2 \
  --gpus-per-node 8 \
  --ntasks-per-node 1 \
  --time 03:00:00 \
  --image verlai/verl:vllm020.dev1 \
  --job-name workload-1k-osl \
  --id exp_rollout_perf_workload_1k_osl_2n8g_YYYYMMDD_HHMM \
  --script /lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/05_claude_ws/exp_rollout_perf_trace_retool_workload_1k_osl/scripts/run_workload_1k_osl.sh
```

The script performs the training run and then exports:

```bash
python3 -m verl.utils.rollout_perf.export_perfetto \
  --input "${PERF_TRACE_DIR}" \
  --output "${PERFETTO_TRACE}"

python3 -m verl.utils.rollout_perf.export_perfetto \
  --input "${PERF_TRACE_DIR}" \
  --output "${PERFETTO_ACTIVE_COUNTER_TRACE}" \
  --mode active_counters \
  --sample-interval-ms 100
```

On a login node without the full verl runtime dependencies, use the file path
instead of `python3 -m` to avoid importing `verl.__init__`:

```bash
cd /path/to/verl
python3 verl/utils/rollout_perf/export_perfetto.py \
  --input /path/to/rollout_perf_trace/JOB_ID \
  --output /path/to/perfetto_active_counters.json \
  --mode active_counters \
  --sample-interval-ms 100
```

Quick syntax check:

```bash
python3 -m py_compile \
  verl/utils/rollout_perf/export_perfetto.py \
  verl/utils/rollout_perf/vllm_metrics.py
```

## Raw Trace Records

All records are JSONL. Each process writes its own file:

```text
rollout_perf_<run_id>_<role>_<hostname>_<pid>_<uuid>.jsonl
```

Common fields added by `writer.py`:

```text
schema_version
record_id
time_unix_ns
time_monotonic_ns
hostname
pid
role
run_id
record_type
name
context
payload
priority
```

Span records also include:

```text
start_unix_ns
end_unix_ns
start_monotonic_ns
end_monotonic_ns
duration_ns
status
```

Current span names:

| Span | Producer | Meaning |
|---|---|---|
| `trajectory` | `agent_loop_worker` | One sampled trajectory from initial prompt to final agent result. |
| `llm_turn` | `agent_loop_worker` | One assistant generation turn in the tool-agent state machine. |
| `llm_client_request` | rollout LLM client | Logical request from agent loop to a selected vLLM server. |
| `tool_turn` | `agent_loop_worker` | Tool-processing state for one turn, often containing multiple tool calls. |
| `tool_call` | `agent_loop_worker` | One concrete tool invocation. |
| `vllm_engine_request` | `vllm_server` | One request handled by a vLLM async server and forwarded to vLLM. |

Important join IDs:

```text
trajectory_id
turn_index
turn_kind
tool_call_id
logical_request_id
engine_request_id
replica_rank
node_rank
engine_index
```

## Perfetto Export Modes

Full mode:

```bash
python3 verl/utils/rollout_perf/export_perfetto.py \
  --input /path/to/raw_trace_dir \
  --output /path/to/perfetto_full.json
```

This exports raw spans, events, and counters as close to the source records as
possible.

Focused counter mode:

```bash
python3 verl/utils/rollout_perf/export_perfetto.py \
  --input /path/to/raw_trace_dir \
  --output /path/to/perfetto_active_counters.json \
  --mode active_counters \
  --sample-interval-ms 100
```

This derives compact counter tracks:

```text
global/agent_loop_worker
agent_loop_worker/a<N>
global/vllm_server
vllm_server/s<N>
global/vllm_engine
vllm_engine/s<N>e<M>
```

Stable IDs are generated from `(hostname, pid)` source keys. They are meant for
readability inside one exported trace, not as durable IDs across runs.

## Counter Naming Rules

Per-worker/per-server counters have only the time-window statistic:

```text
window_max
window_avg
window_p95
```

Global counters use:

```text
window_<time_stat>_<global_agg>
```

Examples:

```text
window_max_sum
window_avg_sum
window_avg_mean
window_avg_max
window_avg_weighted_mean
```

The metric name should describe the measured object. The numeric arg key should
describe how the value was calculated.

## Agent And Server Active Counters

These are derived offline from span start/end intervals.

Per worker/server:

| Track process | Counter name | Args | Meaning |
|---|---|---|---|
| `agent_loop_worker/a<N>` | `active/trajectory` | `window_max` | Max active trajectories on that worker in the sample window. |
| `agent_loop_worker/a<N>` | `active/llm_turn` | `window_max` | Max active LLM turns on that worker in the sample window. |
| `agent_loop_worker/a<N>` | `active/llm_client_request` | `window_max` | Max active logical LLM client requests on that worker in the sample window. |
| `agent_loop_worker/a<N>` | `active/tool_turn` | `window_max` | Max active tool-processing turns on that worker in the sample window. |
| `agent_loop_worker/a<N>` | `active/tool_call` | `window_max` | Max active concrete tool calls on that worker in the sample window. |
| `vllm_server/s<N>` | `active/vllm_engine_request` | `window_max` | Max active outer vLLM server requests in the sample window. |

Global:

| Track process | Counter name | Args | Meaning |
|---|---|---|---|
| `global/agent_loop_worker` | `active/*` | `window_max_sum` | Max active count across all agent workers in the sample window. |
| `global/vllm_server` | `active/vllm_engine_request` | `window_max_sum` | Max active outer vLLM requests across all vLLM servers in the sample window. |

The timestamp of these events is the end of the sampling window. The value
describes the preceding window, not an instantaneous point sample.

## vLLM Engine Counters

vLLM counters come from two sources:

1. vLLM `StatLoggerBase` hook in `RolloutPerfVLLMStatLogger`.
2. Server-side request progress sampler in `RolloutPerfRequestKVMetricsSampler`.

The implementation does not modify vLLM internals. It uses vLLM's stat logger
interface and defensive attribute access because vLLM scheduler/iteration stats
are not a stable public API.

The online vLLM coalescer records all observations inside each engine sampling
window:

```text
raw payload value      = arithmetic mean in the window
raw payload max_value  = max observed value in the window
raw payload num_samples = number of observations in the window
```

Engine metrics are emitted only while that vLLM server has active rollout
requests. When the final local rollout request finishes, pending windows are
flushed and final zero counters are emitted.

Per-engine Perfetto counters:

| Counter name | Args | Raw source | Calculation |
|---|---|---|---|
| `vllm/requests_running` | `window_avg` | `scheduler_stats.num_running_reqs` | Average running requests over the engine sampling window. |
| `vllm/requests_waiting` | `window_avg` | `scheduler_stats.num_waiting_reqs` | Average waiting requests over the engine sampling window. |
| `vllm/requests_batch` | `window_avg` | derived iteration stats | `prefill_requests + decode_requests`, averaged over the window. |
| `vllm/requests_prefill` | `window_avg` | iteration stats | Average number of requests getting first token in the window. |
| `vllm/requests_prefill` | `window_max` | raw `max_value` | Max prefill requests observed in the window. Useful because prefill is sparse. |
| `vllm/requests_decode` | `window_avg` | iteration stats | Average number of decoding requests in the window. |
| `vllm/tokens_batch` | `window_avg` | derived iteration stats | `prefill_tokens_computed + decode_tokens`, averaged over the window. |
| `vllm/tokens_prefill` | `window_avg` | `prompt_token_stats.computed` or fallback total | Average prefill tokens computed in the window. |
| `vllm/tokens_prefill` | `window_max` | raw `max_value` | Max prefill tokens observed in the window. |
| `vllm/tokens_decode` | `window_avg` | derived iteration stats | Average decode tokens in the window. |
| `vllm/kv_cache_usage_%` | `window_avg` | `scheduler_stats.kv_cache_usage` | Window-average KV cache usage ratio multiplied by 100. |
| `vllm/kv_len_per_request_logical` | `window_avg` | server request sampler | Mean logical length of active requests. |
| `vllm/kv_len_per_request_logical` | `window_p95` | server request sampler | Nearest-rank p95 logical length of active requests on that server. |
| `vllm/kv_len_per_request_alloc_est` | `window_avg` | engine KV usage and running requests | Average physical allocated KV tokens per running request estimate. |

Iteration-stat formulas:

```text
prefill_requests = len(iteration_stats.time_to_first_tokens_iter) or 0
decode_requests = len(iteration_stats.inter_token_latencies_iter) or 0
generation_tokens = iteration_stats.num_generation_tokens or 0
decode_tokens = max(generation_tokens - prefill_requests, 0)
prefill_tokens_computed = prompt_token_stats.computed, else prompt_token_stats.total, else 0
batch_requests_total = prefill_requests + decode_requests
batch_tokens_total = prefill_tokens_computed + decode_tokens
```

Logical KV length:

```text
logical_len_per_request = prompt_token_count + cumulative_decoded_tokens
window_avg = mean(logical_len_per_request for active requests)
window_p95 = nearest-rank p95(logical_len_per_request for active requests)
```

Allocated KV length estimate:

```text
allocated_tokens_total = kv_cache_usage_ratio * num_gpu_blocks * block_size
kv_len_per_request_alloc_est = allocated_tokens_total / requests_running
```

This is a global physical allocation estimate divided by running request count.
It is not a true per-request KV block measurement. Therefore V0 exports only
`window_avg` for `kv_len_per_request_alloc_est`; p95 is intentionally omitted.

Global vLLM engine counters:

| Counter name | Args | Meaning |
|---|---|---|
| `vllm/requests_running` | `window_avg_sum` | Sum of per-engine average running requests. |
| `vllm/requests_waiting` | `window_avg_sum` | Sum of per-engine average waiting requests. |
| `vllm/requests_batch` | `window_avg_sum` | Sum of per-engine average batch request count. |
| `vllm/requests_prefill` | `window_avg_sum` | Sum of per-engine average prefill request count. |
| `vllm/requests_prefill` | `window_max_sum` | Sum of per-engine max prefill request count in the window. |
| `vllm/requests_decode` | `window_avg_sum` | Sum of per-engine average decode request count. |
| `vllm/tokens_batch` | `window_avg_sum` | Sum of per-engine average batch tokens. |
| `vllm/tokens_prefill` | `window_avg_sum` | Sum of per-engine average prefill tokens. |
| `vllm/tokens_prefill` | `window_max_sum` | Sum of per-engine max prefill tokens in the window. |
| `vllm/tokens_decode` | `window_avg_sum` | Sum of per-engine average decode tokens. |
| `vllm/kv_cache_usage_%` | `window_avg_mean` | Mean of per-engine KV cache usage percent. |
| `vllm/kv_cache_usage_%` | `window_avg_max` | Max of per-engine KV cache usage percent. |
| `vllm/kv_len_per_request_logical` | `window_avg_weighted_mean` | Weighted mean by `vllm/requests_running {window_avg}`. |
| `vllm/kv_len_per_request_alloc_est` | `window_avg_weighted_mean` | Weighted mean by `vllm/requests_running {window_avg}`. |

Global p95 for KV length is not exported. Per-engine p95 values cannot be
combined into a correct global p95 without a histogram, tdigest, or per-request
distribution samples.

## Validation Commands

Export the current focused trace:

```bash
cd /lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/05_claude_ws/exp_rollout_perf_trace_retool_workload_1k_osl/verl

python3 verl/utils/rollout_perf/export_perfetto.py \
  --input /lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/05_claude_ws/exp_rollout_perf_trace_retool_workload_1k_osl/rollout_perf_trace/12465958 \
  --output /lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/05_claude_ws/exp_rollout_perf_trace_retool_workload_1k_osl/rollout_perf_trace/perfetto_active_counters_global_window_args_12465958.json \
  --mode active_counters \
  --sample-interval-ms 100
```

Check global counter semantics:

```python
import json
from collections import Counter, defaultdict

path = "perfetto_active_counters_global_window_args_12465958.json"
events = json.load(open(path))["traceEvents"]
process_names = {}
for event in events:
    if event.get("ph") == "M" and event.get("name") == "process_name":
        process_names[event["pid"]] = (event.get("args") or {}).get("name")

args_by_process = defaultdict(Counter)
for event in events:
    if event.get("ph") != "C":
        continue
    process = process_names.get(event.get("pid"))
    for key in event.get("args") or {}:
        args_by_process[process][key] += 1

print("global/agent_loop_worker", dict(args_by_process["global/agent_loop_worker"]))
print("global/vllm_server", dict(args_by_process["global/vllm_server"]))
print("global/vllm_engine", dict(args_by_process["global/vllm_engine"]))
```

Expected shape:

```text
global/agent_loop_worker {'window_max_sum': ...}
global/vllm_server {'window_max_sum': ...}
global/vllm_engine {
  'window_avg_mean': ...,
  'window_avg_max': ...,
  'window_avg_weighted_mean': ...,
  'window_avg_sum': ...,
  'window_max_sum': ...
}
```

Check raw record counts:

```python
import json
from collections import Counter
from pathlib import Path

trace_dir = Path("/path/to/rollout_perf_trace/JOB_ID")
counts = Counter()
for path in trace_dir.glob("rollout_perf_*.jsonl"):
    for line in path.open():
        record = json.loads(line)
        counts[(record.get("record_type"), record.get("name"))] += 1
print(counts)
```

## Development Notes

- Prefer keeping raw JSONL complete and format-neutral. Perfetto should remain
  an offline export target, not the stored source format.
- The vLLM integration intentionally uses `StatLoggerBase`; do not patch vLLM
  scheduler/cache internals for V0.
- `rollout.disable_log_stats=True` can make vLLM request metrics unavailable.
  The server logs a warning when engine-internal perf trace is enabled under
  that setting. For richer vLLM metrics, use `disable_log_stats=False`.
- `requests_batch` and token batch metrics are vLLM iteration work, not outer
  request concurrency. Compare them with `active/vllm_engine_request` carefully.
- `active/vllm_engine_request` is outer server request concurrency. It should
  usually be smaller than scheduler batch request count under multi-step decode.
- The focused exporter aligns engine samples to vLLM server active spans and
  zeroes counters outside rollout activity.
- Agent/server active counters are derived from span intervals. They are
  window maxima, not instantaneous point samples.
- Engine counters are online window averages with raw `max_value` metadata.
  Prefill counters export both avg and max because prefill events are sparse.
- The server-side logical KV sampler uses its own thread and interval. It is
  low-cost but not phase-locked with vLLM StatLogger sampling.
- Perfetto JSON counters with multiple numeric args are displayed as separate
  rows in the UI. V0 keeps one numeric arg per counter event and encodes
  semantics in the arg key.
- Current timestamps use Unix ns for cross-process timeline placement and
  monotonic ns for durations. There is no explicit multi-node clock correction.

DFW workflow notes:

- Use `gpu-ssh dfw` and `dfw-run` helpers rather than raw SSH/Slurm commands.
- When uploading files through `gpu-ssh` in a pipeline, redirect each `gpu-ssh`
  invocation from `/dev/null`; otherwise the helper can consume pipeline stdin.
- The validated image is `verlai/verl:vllm020.dev1`.

## Known Limitations And Next Work

V0 limitations:

- No closed-loop replay yet.
- No SGLang/TRT-LLM adapters yet.
- No exact global KV p95.
- No true per-request physical KV block allocation from vLLM internals.
- No explicit timeline track for every vLLM scheduler iteration span.
- No clock synchronization correction across nodes.
- Content capture remains disabled by default; replay will need safe token
  length records, not raw private payloads.

Recommended next steps:

1. Add focused unit tests for `export_perfetto.py` metric naming and aggregation.
2. Add a small raw-trace fixture that covers global agent/server/engine counters.
3. Start P1 closed-loop replay with vLLM as the first engine:
   - load trace-derived prompt/output lengths,
   - drive real engine requests,
   - replace tool calls with sleep plus mock output lengths,
   - compare engine throughput and queue/batch/KV counters under optimized code.
4. Keep replay engine adapters modular so SGLang and TRT-LLM can be added later.
5. Revisit true physical KV allocation if vLLM exposes stable per-request block
   metadata or if we decide to add a deeper optional vLLM adapter.
