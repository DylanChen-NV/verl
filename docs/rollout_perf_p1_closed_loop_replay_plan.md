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
