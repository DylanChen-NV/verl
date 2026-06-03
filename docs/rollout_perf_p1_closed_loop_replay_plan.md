# Rollout Perf P1 Closed-loop Replay Plan

This document records the P1 plan for closed-loop replay. P0 collection and
Perfetto visualization are documented in
`docs/rollout_perf_v0_implementation_guide.md`.

## Goal

Replay a recorded rollout workload with the same dependency structure as the
original trajectory:

```text
initial prompt -> LLM turn -> tool calls -> next LLM turn -> ... -> final output
```

The replay should drive a real inference engine for LLM requests, replace tool
execution with sleep plus mock output length, and emit a new rollout perf trace.
The main use case is evaluating inference-engine optimizations against a
production-like workload without needing to run real tools or reward logic.

## Acceptance Criteria

P1 is considered useful when these conditions hold:

1. A raw trace can be converted into a deterministic `ReplayPlan`.
2. A mock engine can execute the plan end-to-end without GPUs.
3. A vLLM engine can execute the same plan with synthetic prompts and target
   output lengths.
4. Replay emits raw rollout perf records and can be exported by the existing
   Perfetto exporter.
5. With the same engine/image/resources and no engine-side code change, original
   and replay engine-side metrics are close enough to trust replay for
   optimization studies.

## Fidelity Test

The final end-to-end test compares:

```text
original raw trace -> original summary / active_counters
replay raw trace   -> replay summary / active_counters
```

Metrics to compare:

```text
total trajectory count
total LLM request count
total generated tokens
total tool call count
total tool sleep time
total wall time
global/vllm_engine request and token counters
global/vllm_engine KV cache and KV length counters
global/agent_loop_worker active counters
global/vllm_server active counters
```

Initial thresholds should be broad and tightened after the first vLLM replay:

```text
LLM request count: exact
generated tokens: exact or <1% delta
tool call count: exact
tool sleep aggregate: <5% delta
wall time: <20% delta
engine batch/token aggregates: <20-30% delta
KV cache avg/max: <20-30% delta
```

The comparison should not require timestamp-by-timestamp equality. Closed-loop
replay intentionally lets engine latency changes affect later turn arrival
times.

## Replay Semantics

Source of truth for each LLM turn:

```text
llm_turn span
llm_client_request span
vllm_engine_request span
```

The plan builder joins these by:

```text
trajectory_id
turn_index
logical_request_id
engine_request_id
```

Source of truth for tools:

```text
tool_turn span
tool_call span
```

Within one tool turn, tool calls should run concurrently and join before the
next LLM turn. Tool calls sleep for the recorded duration and return a mock
output with the recorded token length.

Closed-loop rule:

- The first trajectory starts according to the replay arrival policy.
- Within a trajectory, the next operation starts only after the previous replay
  operation finishes.
- Later turns are not triggered by the original absolute timestamps.

## ReplayPlan Format

The first plan format is JSON and stores lengths, timings, IDs, and dependency
structure. It intentionally avoids raw prompt/tool content.

```text
ReplayPlan
  version
  metadata
  trajectories[]

ReplayTrajectory
  trajectory_id
  global_step
  sample_index
  rollout_n
  validate
  agent_name
  start_offset_ns
  original_duration_ns
  prompt_token_count
  response_token_count
  llm_generated_token_count
  tool_response_token_count
  tool_call_count
  turns[]

ReplayTurn
  turn_index
  llm_request
  tool_turn_duration_ns
  tool_response_token_count
  tool_calls[]

ReplayLLMRequest
  logical_request_id
  engine_request_id
  prompt_token_count
  output_token_count
  requested_max_tokens
  original_llm_duration_ns
  original_client_duration_ns
  original_engine_duration_ns
  original_ttft_ns
  stop_reason
  status

ReplayToolCall
  tool_call_id
  tool_name
  argument_token_count
  output_token_count
  sleep_ns
  status
```

## Implementation Phases

### Phase 1: Plan and Mock Replay

Add:

```text
verl/utils/rollout_perf/replay/schema.py
verl/utils/rollout_perf/replay/trace_loader.py
verl/utils/rollout_perf/replay/plan_builder.py
verl/utils/rollout_perf/replay/replay_writer.py
verl/utils/rollout_perf/replay/tool_mock.py
verl/utils/rollout_perf/replay/runner.py
verl/utils/rollout_perf/replay/compare.py
verl/utils/rollout_perf/replay/engines/base.py
verl/utils/rollout_perf/replay/engines/mock.py
```

Deliverables:

- Build `replay_plan.json` from the validated ReTool trace.
- Run a no-GPU mock replay.
- Emit replay raw trace records.
- Produce a basic original-vs-replay comparison report.

### Phase 2: vLLM Single-node Smoke

Add:

```text
verl/utils/rollout_perf/replay/engines/vllm.py
```

The vLLM adapter should synthesize prompts by token length and request exact
output lengths:

```text
SamplingParams(max_tokens=target_output_tokens, ignore_eos=True, temperature=0.0)
```

Validation questions:

- Does vLLM v0.20.2 reliably emit exactly `max_tokens` with `ignore_eos=True`?
- What should happen when `prompt_token_count + output_token_count` exceeds
  `max_model_len`?
- Can the adapter reuse existing verl/vLLM server setup instead of duplicating
  distributed bootstrap logic?

### Phase 3: 2-node Fidelity Test

Run replay using the same image, model, and comparable DFW resources as the
source trace:

```text
image: verlai/verl:vllm020.dev1
model: Qwen2.5-7B-Instruct
nodes: 2
gpus_per_node: 8
```

Export original and replay focused active counters, then run compare. If the
engine-side metrics are too different without code changes, fix replay before
using it for engine optimization evaluation.

## CLI

Build a plan:

```bash
python3 -m verl.utils.rollout_perf.replay.plan_builder \
  --input /path/to/rollout_perf_trace/JOB_ID \
  --output /path/to/replay_plan.json
```

Run mock replay:

```bash
python3 -m verl.utils.rollout_perf.replay.runner \
  --plan /path/to/replay_plan.json \
  --engine mock \
  --output-dir /path/to/replay_trace_mock \
  --arrival-mode original \
  --time-scale 1.0
```

Fast no-sleep dry-run:

```bash
python3 -m verl.utils.rollout_perf.replay.runner \
  --plan /path/to/replay_plan.json \
  --engine mock \
  --output-dir /path/to/replay_trace_mock_fast \
  --arrival-mode burst \
  --time-scale 0.0
```

Compare:

```bash
python3 -m verl.utils.rollout_perf.replay.compare \
  --original /path/to/original_trace \
  --replay /path/to/replay_trace \
  --output /path/to/replay_compare_report.md
```

The modules also support direct file execution on login nodes where
`python3 -m verl...` would import the top-level `verl` package and require full
runtime dependencies.

## Open Issues

- Exact output length control must be validated with vLLM.
- Multi-node vLLM replay bootstrap is expected to be the largest P1 engineering
  item.
- Current raw trace does not store raw token IDs. That is intentional for V0,
  but replay therefore validates workload shape and lengths, not content.
- Global KV p95 remains out of scope until the collector stores a mergeable
  distribution.
- Per-request physical KV allocation remains out of scope unless vLLM exposes a
  stable API or an optional deeper adapter is added.

## Implementation Status: 2026-06-03

P1 closed-loop replay has a first complete implementation on branch
`rollout-perf-tool-dev`.

Implemented modules:

```text
verl/utils/rollout_perf/replay/schema.py
verl/utils/rollout_perf/replay/trace_loader.py
verl/utils/rollout_perf/replay/plan_builder.py
verl/utils/rollout_perf/replay/replay_writer.py
verl/utils/rollout_perf/replay/tool_mock.py
verl/utils/rollout_perf/replay/runner.py
verl/utils/rollout_perf/replay/compare.py
verl/utils/rollout_perf/replay/engines/base.py
verl/utils/rollout_perf/replay/engines/mock.py
verl/utils/rollout_perf/replay/engines/vllm.py
```

Validated source trace and replay plan:

```text
original trace job: 12465958
original trace dir:
/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/05_claude_ws/exp_rollout_perf_trace_retool_workload_1k_osl/rollout_perf_trace/12465958

replay plan:
/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/05_claude_ws/exp_rollout_perf_trace_retool_workload_1k_osl/rollout_perf_trace/replay_plan_12465958.json

plan size:
trajectories=96
llm_requests=490
tool_calls=394
generated_tokens=32747
tool_output_tokens=39549
max_prompt_plus_output_tokens=2953
```

Validated image and model:

```text
image sqsh:
/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/05_claude_ws/public_images/verl/verlai-verl-vllm020.dev1.sqsh

source docker ref:
verlai/verl:vllm020.dev1

vLLM version: 0.20.2
model:
/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/05_claude_ws/public_models/verl/Qwen2.5-7B-Instruct
```

Docker-ref startup note:

```text
Direct Pyxis startup from verlai/verl:vllm020.dev1 failed once after image
conversion. A stable sqsh was created with srun --container-save and should be
used for replay validation.
```

Exact output length validation:

```text
DFW job: 12472527
run id: rollout_perf_vllm_len_smoke_20260603_095915
target lengths: 1, 8, 32, 128
observed lengths: exact
```

The vLLM adapter requests exact decode length with:

```python
SamplingParams(
    max_tokens=output_len,
    min_tokens=output_len,
    ignore_eos=True,
    temperature=0.0,
    detokenize=False,
)
```

Synthetic prompts are generated by repeating a stable tokenizer token to match
`prompt_token_count`. Replay validates workload shape, token lengths, batching,
KV behavior, and timing pressure. It does not validate prompt content semantics.

vLLM internal metric integration detail:

```text
vLLM 0.20.2 LLM(...) does not accept stat_loggers directly; that kwarg is
forwarded to EngineArgs and fails. The replay adapter therefore constructs
LLM(disable_log_stats=False), then installs a StatLoggerManager on
llm.llm_engine with RolloutPerfVLLMStatLogger before any generation request.
```

Replay-specific environment variables:

```text
TMPDIR=/tmp/rollout_perf_replay_<job_id>
VLLM_RPC_BASE_PATH=${TMPDIR}/rpc
VERL_ROLLOUT_PERF_OUTPUT_DIR=<replay output dir>
VERL_ROLLOUT_PERF_ENGINE_INTERNAL_SAMPLE_INTERVAL_MS=100
VERL_ROLLOUT_PERF_FORCE_ENGINE_INTERNAL_ACTIVE=1
VERL_ROLLOUT_PERF_FLUSH_INTERVAL_S=1.0
```

`VERL_ROLLOUT_PERF_FORCE_ENGINE_INTERNAL_ACTIVE=1` is only for standalone
replay. In normal verl rollout, vLLM internal sampling is gated by rollout
activity from the server wrapper.

### Validated Runs

No-GPU mock replay:

```text
output dir:
/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/05_claude_ws/exp_rollout_perf_trace_retool_workload_1k_osl/rollout_perf_trace/replay_mock_full_fast

compare report:
/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/05_claude_ws/exp_rollout_perf_trace_retool_workload_1k_osl/rollout_perf_trace/replay_compare_mock_full_fast.md

Perfetto active counters:
/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/05_claude_ws/exp_rollout_perf_trace_retool_workload_1k_osl/rollout_perf_trace/perfetto_active_counters_replay_mock_full_fast.json
```

1GPU vLLM smoke with internal counters:

```text
DFW job: 12473415
trajectories=2
engine_requests=11
generated_tokens=824
tool_calls=9

output dir:
/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/05_claude_ws/exp_rollout_perf_trace_retool_workload_1k_osl/rollout_perf_trace/replay_vllm_smoke_1g_12473415

compare report:
/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/05_claude_ws/exp_rollout_perf_trace_retool_workload_1k_osl/rollout_perf_trace/replay_compare_vllm_smoke_1g_12473415_counters.md

Perfetto active counters:
/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/05_claude_ws/exp_rollout_perf_trace_retool_workload_1k_osl/rollout_perf_trace/perfetto_active_counters_replay_vllm_smoke_1g_12473415.json
```

Full 1GPU closed-loop replay:

```text
DFW job: 12473542
arrival_mode=original
time_scale=1
max_concurrency=96
max_model_len=4096
vllm_sample_interval_ms=100

output dir:
/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/05_claude_ws/exp_rollout_perf_trace_retool_workload_1k_osl/rollout_perf_trace/replay_vllm_full_1g_12473542

compare markdown:
/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/05_claude_ws/exp_rollout_perf_trace_retool_workload_1k_osl/rollout_perf_trace/replay_compare_vllm_full_1g_12473542.md

compare json:
/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/05_claude_ws/exp_rollout_perf_trace_retool_workload_1k_osl/rollout_perf_trace/replay_compare_vllm_full_1g_12473542.json

Perfetto active counters:
/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/05_claude_ws/exp_rollout_perf_trace_retool_workload_1k_osl/rollout_perf_trace/perfetto_active_counters_replay_vllm_full_1g_12473542.json
```

Full replay comparison result:

```text
trajectories: original=96, replay=96
engine_requests: original=490, replay=490
engine_output_tokens: original=32747, replay=32747
tool_calls: original=394, replay=394
tool_sleep_ms: original=4800.65, replay=5142.94, delta=+7.1%
wall_time_ms: original=40345.23, replay=49181.51, delta=+21.9%
```

The full replay raw trace contained:

```text
counter=5952
span=2354
event=3
run_manifest=2
```

The remaining engine metric differences are expected for this validation because
it replays a 2-node/8-GPU source workload on a single GPU and uses
`--enforce-eager` for stable smoke-style startup. For engine optimization
studies, run replay with the target engine build and target resource shape.

### Re-run Commands

Build the replay plan:

```bash
cd /lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/05_claude_ws/exp_rollout_perf_trace_retool_workload_1k_osl/verl
python3 verl/utils/rollout_perf/replay/plan_builder.py \
  --input ../rollout_perf_trace/12465958 \
  --output ../rollout_perf_trace/replay_plan_12465958.json
```

Run full 1GPU replay through the saved script:

```bash
/home/scratch.ziqingc_gpu/06_codex_projects/01_template/gpu-workflow/bin/dfw-run submit \
  --nodes 1 \
  --gpus-per-node 1 \
  --ntasks-per-node 1 \
  --time 02:00:00 \
  --image /lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/05_claude_ws/public_images/verl/verlai-verl-vllm020.dev1.sqsh \
  --job-name replay-vllm-full \
  --id rollout_perf_replay_vllm_full_1g_YYYYMMDD_HHMMSS \
  --wait \
  --script /lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/05_claude_ws/exp_rollout_perf_trace_retool_workload_1k_osl/scripts/run_vllm_replay_full_1g.sh
```

Run the runner directly inside the container:

```bash
python3 verl/utils/rollout_perf/replay/runner.py \
  --plan "${PLAN}" \
  --engine vllm \
  --model "${MODEL}" \
  --output-dir "${OUT}" \
  --arrival-mode original \
  --time-scale 1 \
  --max-concurrency 96 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.75 \
  --vllm-batch-wait-ms 5 \
  --vllm-max-batch-size 32 \
  --vllm-sample-interval-ms 100 \
  --enforce-eager
```

Compare and export Perfetto:

```bash
python3 verl/utils/rollout_perf/replay/compare.py \
  --original "${ORIGINAL_TRACE_DIR}" \
  --replay "${OUT}" \
  --output "${COMPARE_MD}" \
  --json-output "${COMPARE_JSON}"

python3 verl/utils/rollout_perf/export_perfetto.py \
  --input "${OUT}" \
  --output "${PERFETTO}" \
  --mode active_counters \
  --sample-interval-ms 100
```

`compare.py` now includes a counter summary. It summarizes each raw counter name
by samples, mean, p50, p95, p99, max, and sum over `payload.value`. If
`payload.max_value` exists, those stats are included in the JSON report.

### Current Limitations And TODO

- Multi-node standalone vLLM replay remains a TODO. The P1 validation above is
  full-workload but single-node/single-GPU.
- Replay uses synthetic repeated-token prompts. This is intentional for P1 and
  keeps content out of the trace.
- Closed-loop replay reproduces dependency structure and lengths, not absolute
  per-turn timestamps. Engine latency changes intentionally affect later turns.
- The source trace still contains `vllm/kv_len_per_request_alloc_est_p95` from an
  older counter set; current replay no longer emits that counter because alloc
  length p95 was intentionally removed.
- Open-loop replay and multi-engine adapters for SGLang/TRT-LLM remain P2 TODOs.
