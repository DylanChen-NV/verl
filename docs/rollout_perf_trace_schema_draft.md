# Rollout Perf Trace Schema Draft

Purpose: one raw trace format that can support rollout collection, timeline
visualization, workload analysis, and closed-loop replay. This is a draft for
review comments.

Priority meaning:

```text
P0 = required for the first implementation target: collection + visualization,
     including vLLM internal state needed to understand batching/KV/scheduler
     behavior. Some P0 fields also make closed-loop replay possible later.
P1 = useful enrichment, deeper diagnosis, privacy-sensitive payload, or fields
     that can be derived offline from P0 data.
```

## Design Notes

The trace should be stored as append-only records, preferably JSONL or msgpack
records, with periodic manifest/checkpoint records. Perfetto export should be an
offline conversion from this raw trace, not the primary persisted format.

Every record should carry enough IDs to join four views:

```text
step -> trajectory -> turn -> llm_request/tool_call
worker/process/thread -> timeline_event
engine_worker -> scheduler_iteration -> llm_request
```

## Common Fields For Every Record

| Field | Priority | Type | Description |
|---|---:|---|---|
| schema_version | P0 | string | Trace schema version, for example `rollout-perf/v0.1`. |
| record_type | P0 | enum | One of `run_manifest`, `step`, `trajectory`, `turn`, `llm_request`, `tool_call`, `engine_scheduler`, `worker_timeline`, `summary`, `error`. |
| trace_id | P0 | string | Unique trace file/session id. |
| run_id | P0 | string | Training or smoke run id. |
| global_step | P0 | int/null | Trainer global step if known. |
| timestamp_ns | P0 | int | Monotonic or wall-clock timestamp in ns; source must be declared in manifest. |
| wall_time_ms | P1 | int | Unix epoch milliseconds, useful for cross-log correlation. |
| producer | P0 | string | Component that emitted the record, for example `verl.agent_loop`, `vllm_adapter`, `tool_wrapper`. |
| producer_host | P0 | string | Hostname or node id. |
| producer_pid | P0 | int | Process id. |
| producer_thread_id | P1 | int/null | Thread id if available. |
| sequence_no | P0 | int | Per-producer monotonically increasing sequence number. |
| parent_record_id | P1 | string/null | Optional causal parent record id. |
| record_id | P0 | string | Unique id for this record. |

## Run Manifest

| Field | Priority | Type | Description |
|---|---:|---|---|
| experiment_name | P0 | string | Trainer experiment name. |
| project_name | P0 | string | Trainer/W&B project name. |
| wandb_url | P1 | string/null | W&B run URL if enabled. |
| git_commit | P0 | string | verl commit. |
| git_branch | P1 | string | Branch name. |
| git_dirty | P0 | bool | Whether code had uncommitted changes. |
| recipe_commit | P0 | string/null | recipe submodule commit if present. |
| container_image | P0 | string | Docker or sqsh image. |
| verl_version | P0 | string | Runtime verl version. |
| inference_engine | P0 | enum | `vllm`, `sglang`, `trtllm`, `mock`, etc. |
| inference_engine_version | P0 | string | Engine version, for example `vllm=0.20.2`. |
| engine_adapter_version | P0 | string | Rollout perf adapter implementation version. |
| torch_version | P0 | string | Torch version. |
| ray_version | P0 | string | Ray version. |
| cuda_version | P1 | string/null | CUDA runtime/toolkit version. |
| nccl_version | P1 | string/null | NCCL version. |
| model_name_or_path | P0 | string | Model path/name used by rollout. |
| tokenizer_name_or_path | P0 | string | Tokenizer path/name. |
| model_config_hash | P1 | string/null | Hash of relevant model config. |
| trainer_config_hash | P0 | string | Hash of resolved trainer config. |
| trainer_config_inline | P1 | object/string | Full resolved config, if allowed. |
| cluster_name | P0 | string | DFW/OCI/local/etc. |
| slurm_job_id | P0 | string/null | Slurm job id if present. |
| node_count | P0 | int | Allocated node count. |
| gpus_per_node | P0 | int | GPUs per node. |
| node_names | P0 | list[string] | Allocated node names. |
| gpu_names | P1 | list[string] | GPU model names. |
| worker_topology | P0 | list[object] | Mapping of worker ids to host, pid, rank, GPU ids, engine role. |
| clock_source | P0 | enum | `monotonic_ns`, `perf_counter_ns`, `epoch_ns`, or mixed. |
| content_capture_policy | P0 | object | Whether raw prompts/tool IO/text are captured or redacted. |
| privacy_redaction_version | P1 | string/null | Redaction rule version. |

## Step Record

| Field | Priority | Type | Description |
|---|---:|---|---|
| step_id | P0 | string | Unique step id. |
| epoch | P0 | int | Training epoch. |
| train_batch_size | P0 | int | Prompt batch size. |
| n_responses_per_prompt | P0 | int | Rollout `n`. |
| trajectory_count | P0 | int | Number of sampled trajectories in the step. |
| prompt_count | P0 | int | Number of unique prompts. |
| step_start_ns | P0 | int | Step start. |
| step_end_ns | P0 | int/null | Step end. |
| rollout_start_ns | P0 | int/null | Rollout collection start. |
| rollout_end_ns | P0 | int/null | Rollout collection end. |
| update_start_ns | P1 | int/null | Actor update start. |
| update_end_ns | P1 | int/null | Actor update end. |
| validation_step | P0 | bool | Whether this is validation/inference-only. |
| status | P0 | enum | `running`, `completed`, `failed`, `aborted`. |
| aggregate_prompt_tokens | P0 | int | Total prompt tokens in step. |
| aggregate_response_tokens | P0 | int | Total response tokens in step. |
| aggregate_effective_tokens | P0 | int | Total trajectory tokens after tool results/messages are included. |
| aggregate_tool_calls | P0 | int | Total tool calls in step. |
| aggregate_turns | P0 | int | Total turns in step. |
| throughput_tokens_per_s | P1 | float/null | Derived or logged throughput. |
| step_summary_ref | P1 | string/null | Pointer to offline summary output. |

## Trajectory Record

| Field | Priority | Type | Description |
|---|---:|---|---|
| trajectory_id | P0 | string | Unique trajectory id. |
| step_id | P0 | string | Parent step id. |
| prompt_id | P0 | string | Stable prompt id. |
| sample_index | P0 | int | Response index for prompt when `n > 1`. |
| data_source | P1 | string/null | Dataset/source name. |
| initial_prompt_tokens | P0 | int | Initial user/system prompt token count. |
| initial_prompt_bytes | P1 | int | Initial prompt byte length. |
| final_response_tokens | P0 | int | Final assistant response token count as used by trainer metrics. |
| effective_total_tokens | P0 | int | Tokens in full trajectory context after all turns/tool outputs. |
| generated_tokens_total | P0 | int | Sum of tokens generated by LLM across turns. |
| tool_output_tokens_total | P0 | int | Sum of tool output tokens inserted into context. |
| tool_input_tokens_total | P1 | int | Sum of tool argument/input tokens. |
| num_turns | P0 | int | Total dialogue turns/messages counted by agent loop. |
| num_assistant_turns | P0 | int | Assistant turns. |
| num_user_turns | P0 | int | User/tool-result turns if represented as user messages. |
| num_tool_calls | P0 | int | Number of executed tool calls. |
| max_context_tokens_seen | P0 | int | Max context/KV length seen by LLM for this trajectory. |
| status | P0 | enum | `completed`, `aborted`, `tool_error`, `llm_error`, `timeout`. |
| stop_reason | P0 | string/null | Stop reason from agent loop or engine. |
| reward | P1 | float/null | Final reward/score. |
| score | P1 | float/null | Task score if separate from reward. |
| aborted | P0 | bool | Whether response was aborted/clipped. |
| error_id | P0 | string/null | Link to error record. |
| replay_plan_id | P0 | string | Link to replay plan derived from this trajectory. |
| content_hash | P1 | string/null | Hash of full rendered trajectory content. |
| raw_trajectory_ref | P1 | string/null | Optional external blob ref for full text/token ids. |

## Turn Record

| Field | Priority | Type | Description |
|---|---:|---|---|
| turn_id | P0 | string | Unique turn id. |
| trajectory_id | P0 | string | Parent trajectory. |
| turn_index | P0 | int | Zero-based turn index. |
| turn_kind | P0 | enum | `llm`, `tool`, `system`, `score`, `other`. |
| actor_role | P0 | enum | `assistant`, `user`, `tool`, `system`. |
| input_context_tokens | P0 | int | Tokens in context before this turn action. |
| input_context_bytes | P1 | int | Rendered input bytes before this turn action. |
| output_tokens | P0 | int | Tokens emitted by this turn action. |
| output_bytes | P1 | int | Output bytes. |
| start_ns | P0 | int | Turn start. |
| end_ns | P0 | int | Turn end. |
| llm_request_id | P0 | string/null | LLM request for this turn if any. |
| tool_call_ids | P0 | list[string] | Tool calls made or fulfilled in this turn. |
| message_hash | P1 | string/null | Hash of rendered message. |
| raw_message_ref | P1 | string/null | Optional full content ref. |
| status | P0 | enum | `completed`, `failed`, `skipped`, `timeout`. |

## LLM Request Record

| Field | Priority | Type | Description |
|---|---:|---|---|
| llm_request_id | P0 | string | Unique LLM generation request id. |
| trajectory_id | P0 | string | Parent trajectory. |
| turn_id | P0 | string | Parent turn. |
| engine_request_id | P0 | string | Request id seen by inference engine. |
| engine_type | P0 | enum | `vllm`, `sglang`, `trtllm`, `mock`. |
| engine_worker_id | P0 | string | Engine worker/server id. |
| host | P0 | string | Host handling request. |
| pid | P0 | int | Process handling request. |
| model_name_or_path | P0 | string | Model path/name. |
| sampling_temperature | P0 | float | Temperature. |
| sampling_top_p | P0 | float | Top-p. |
| sampling_top_k | P0 | int | Top-k. |
| sampling_max_tokens | P0 | int | Max generation tokens. |
| sampling_n | P0 | int | Number of responses requested. |
| prompt_tokens | P0 | int | Input tokens to engine. |
| prompt_bytes | P1 | int | Rendered prompt bytes. |
| prompt_token_ids_ref | P1 | string/null | Optional token ids blob ref. |
| generated_tokens | P0 | int | Tokens generated by engine. |
| generated_bytes | P1 | int | Generated text bytes. |
| generated_token_ids_ref | P1 | string/null | Optional generated token ids blob ref. |
| generated_text_hash | P1 | string/null | Hash of generated text. |
| raw_generated_text_ref | P1 | string/null | Optional text blob ref, privacy controlled. |
| arrival_ns | P0 | int | Request emitted by agent/rollout. |
| enqueue_ns | P0 | int/null | Engine queue enqueue time. |
| first_scheduled_ns | P0 | int/null | First scheduler selection time. |
| prefill_start_ns | P0 | int/null | Prefill start. |
| prefill_end_ns | P0 | int/null | Prefill end. |
| decode_start_ns | P0 | int/null | Decode start. |
| first_token_ns | P0 | int/null | First output token time. |
| decode_end_ns | P0 | int/null | Decode end. |
| complete_ns | P0 | int | Request completion time. |
| queue_time_ns | P0 | int/null | Engine queue duration. |
| prefill_time_ns | P0 | int/null | Prefill duration. |
| decode_time_ns | P0 | int/null | Decode duration. |
| time_to_first_token_ns | P0 | int/null | TTFT. |
| tokens_per_s | P0 | float/null | Per-request generation throughput. |
| stop_reason | P0 | string/null | Engine stop reason. |
| finish_reason | P0 | string/null | OpenAI/vLLM finish reason if available. |
| aborted | P0 | bool | Whether engine aborted request. |
| num_preemptions | P0 | int/null | Number of preemptions, if available. |
| prefix_cache_hit_tokens | P0 | int/null | Prefix cache hit tokens. |
| kv_cache_tokens_allocated | P0 | int/null | KV/cache tokens allocated to request. |
| scheduler_iteration_ids | P0 | list[string] | Scheduler iterations that touched this request. |
| batch_ids | P0 | list[string] | Batches that included this request. |

## Tool Call Record

| Field | Priority | Type | Description |
|---|---:|---|---|
| tool_call_id | P0 | string | Unique tool call id. |
| trajectory_id | P0 | string | Parent trajectory. |
| turn_id | P0 | string | Parent turn. |
| llm_request_id | P0 | string/null | LLM request that produced the call. |
| tool_name | P0 | string | Tool name, for example `code_interpreter`. |
| tool_worker_id | P0 | string | Tool process/worker id. |
| host | P0 | string | Tool host. |
| pid | P0 | int | Tool process id. |
| call_index_in_turn | P0 | int | Order within a turn. |
| args_tokens | P0 | int | Token count for tool arguments. |
| args_bytes | P0 | int | Byte count for tool arguments. |
| args_hash | P1 | string/null | Hash of arguments. |
| raw_args_ref | P1 | string/null | Optional raw args blob ref. |
| output_tokens | P0 | int | Token count of tool output inserted into trajectory. |
| output_bytes | P0 | int | Byte count of tool output inserted into trajectory. |
| output_hash | P1 | string/null | Hash of tool output. |
| raw_output_ref | P1 | string/null | Optional raw output blob ref. |
| scheduled_ns | P0 | int | Tool call scheduled time. |
| start_ns | P0 | int | Tool execution start. |
| end_ns | P0 | int | Tool execution end. |
| latency_ns | P0 | int | Tool execution latency. |
| queue_time_ns | P1 | int/null | Time waiting for tool worker. |
| timeout_ms | P0 | int/null | Configured timeout. |
| status | P0 | enum | `success`, `timeout`, `error`, `cancelled`. |
| return_code | P1 | int/null | Tool return code, if relevant. |
| error_id | P0 | string/null | Link to error record. |
| replay_sleep_ns | P0 | int | Sleep duration to use in closed-loop replay. |
| replay_output_tokens | P0 | int | Mock tool output token length for replay. |

## Engine Scheduler / vLLM Internal Record

These records are required for P0 because the current requirement explicitly
includes vLLM internal information. The same shape should be emitted by engine
adapters for SGLang/TRT-LLM where possible.

| Field | Priority | Type | Description |
|---|---:|---|---|
| scheduler_iteration_id | P0 | string | Unique scheduler iteration id. |
| engine_type | P0 | enum | `vllm`, `sglang`, `trtllm`, `mock`. |
| engine_worker_id | P0 | string | Engine worker/server id. |
| host | P0 | string | Host. |
| pid | P0 | int | Process id. |
| gpu_ids | P0 | list[int] | GPUs used by this engine worker. |
| iteration_start_ns | P0 | int | Scheduler iteration start. |
| iteration_end_ns | P0 | int | Scheduler iteration end. |
| scheduled_request_ids | P0 | list[string] | Engine request ids scheduled in this iteration. |
| waiting_request_count | P0 | int | Queue length before/at scheduling. |
| running_request_count | P0 | int | Running requests. |
| swapped_request_count | P0 | int/null | Swapped requests if engine supports it. |
| preempted_request_count | P0 | int/null | Preemptions in iteration. |
| batch_id | P0 | string | Batch id created/updated by scheduler. |
| batch_num_sequences | P0 | int | Number of sequences in batch. |
| batch_total_tokens | P0 | int | Total scheduled tokens in batch. |
| batch_prefill_tokens | P0 | int | Prefill tokens scheduled. |
| batch_decode_tokens | P0 | int | Decode tokens scheduled. |
| max_num_batched_tokens | P0 | int/null | Engine configured token budget. |
| max_num_seqs | P0 | int/null | Engine configured seq budget. |
| kv_cache_total_blocks | P0 | int/null | Total KV blocks. |
| kv_cache_free_blocks | P0 | int/null | Free KV blocks. |
| kv_cache_used_blocks | P0 | int/null | Used KV blocks. |
| kv_cache_usage_ratio | P0 | float/null | Used/total blocks. |
| gpu_cache_usage_ratio | P0 | float/null | vLLM cache usage stat if exposed. |
| cpu_cache_usage_ratio | P1 | float/null | CPU/offload cache usage. |
| prefix_cache_hit_rate | P0 | float/null | Prefix cache hit rate. |
| prefix_cache_hit_tokens | P0 | int/null | Prefix cache hit tokens in iteration/window. |
| num_generation_tokens_emitted | P0 | int | Tokens emitted this iteration/window. |
| num_prompt_tokens_processed | P0 | int | Prompt tokens processed this iteration/window. |
| engine_throughput_tokens_per_s | P0 | float/null | Engine-level throughput snapshot. |
| gpu_memory_used_bytes | P1 | int/null | GPU memory usage. |
| gpu_memory_free_bytes | P1 | int/null | GPU memory free. |
| scheduler_debug_ref | P1 | string/null | Optional raw engine-specific stats blob. |

## Worker Timeline Event

This is the raw form that should export cleanly to Perfetto slices/counters.

| Field | Priority | Type | Description |
|---|---:|---|---|
| timeline_event_id | P0 | string | Unique event id. |
| actor_id | P0 | string | Logical worker actor id. |
| actor_kind | P0 | enum | `agent_loop`, `vllm_server`, `tool_worker`, `trainer`, `ray_worker`, `replay_engine`. |
| host | P0 | string | Host. |
| pid | P0 | int | Process id. |
| thread_id | P1 | int/null | Thread id. |
| gpu_id | P1 | int/null | GPU id if event is GPU-bound. |
| state | P0 | enum | `idle`, `queueing`, `llm_prefill`, `llm_decode`, `llm_mixed`, `tool_call`, `postprocess`, `reward`, `update`, `sync`, `sleep`, `error`. |
| start_ns | P0 | int | Event start. |
| end_ns | P0 | int | Event end. |
| duration_ns | P0 | int | Event duration. |
| trajectory_id | P0 | string/null | Linked trajectory if relevant. |
| turn_id | P0 | string/null | Linked turn if relevant. |
| llm_request_id | P0 | string/null | Linked LLM request if relevant. |
| tool_call_id | P0 | string/null | Linked tool call if relevant. |
| scheduler_iteration_id | P0 | string/null | Linked scheduler iteration if relevant. |
| counter_name | P1 | string/null | For counter events, for example `kv_cache_usage_ratio`. |
| counter_value | P1 | float/null | Counter value. |
| details_ref | P1 | string/null | Optional extra metadata blob. |

## Replay Plan Fields

Closed-loop replay is P1 in development priority, but the P0 trace should already
preserve enough length/timing information to build it.

| Field | Priority | Type | Description |
|---|---:|---|---|
| replay_plan_id | P0 | string | Stable id linked from trajectory. |
| trajectory_id | P0 | string | Source trajectory. |
| replay_engine_type | P1 | enum | Engine used during replay. |
| initial_prompt_tokens | P0 | int | Prompt length for replay request. |
| turn_count | P0 | int | Number of turns to replay. |
| per_turn_llm_input_tokens | P0 | list[int] | LLM input length per turn. |
| per_turn_llm_output_tokens | P0 | list[int] | LLM output length per turn. |
| per_turn_tool_sleep_ns | P0 | list[int] | Mock tool sleep per tool call/turn. |
| per_turn_tool_output_tokens | P0 | list[int] | Mock tool output length per tool call/turn. |
| dependencies | P0 | list[object] | Causal dependencies, for example LLM output -> tool call -> next LLM input. |
| sampling_params | P0 | object | Sampling parameters to reproduce engine load shape. |
| concurrency_group | P1 | string/null | Replay grouping for workload scheduling. |
| original_timing_policy | P1 | enum | `asap`, `preserve_interarrival`, `preserve_turn_gap`. |
| synthetic_content_policy | P1 | object | How to make synthetic token ids/text. |

## Summary / Analysis Output Fields

These can be generated offline from P0 records. Keep them out of the hot path
unless cheap.

| Field | Priority | Type | Description |
|---|---:|---|---|
| summary_window | P1 | object | Step/time/worker window covered by summary. |
| p50_p95_p99_latency | P1 | object | Latency quantiles for LLM/tool/trajectory. |
| token_throughput_by_worker | P1 | object | Per-worker throughput summary. |
| kv_cache_utilization_histogram | P1 | object | KV/cache utilization distribution. |
| batching_histogram | P1 | object | Batch size/token histogram. |
| bottleneck_label | P1 | enum | Offline diagnosis label. |
| critical_path_trajectory_id | P1 | string/null | Slowest/critical trajectory. |
| perfetto_export_path | P1 | string/null | Path to generated Perfetto trace. |

## Error Record

| Field | Priority | Type | Description |
|---|---:|---|---|
| error_id | P0 | string | Unique error id. |
| component | P0 | string | Component that failed. |
| error_type | P0 | string | Exception/error type. |
| error_message | P0 | string | Short message. |
| traceback_hash | P0 | string/null | Hash of traceback. |
| traceback_ref | P1 | string/null | Optional full traceback blob ref. |
| linked_record_ids | P0 | list[string] | Records affected by error. |
| recoverable | P1 | bool/null | Whether component recovered. |
