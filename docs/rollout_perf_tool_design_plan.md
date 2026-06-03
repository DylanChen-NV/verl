# Rollout Perf Tool Design Plan

本文档记录 rollout perf tool 当前调研结论和开发方案。字段级 schema 见
`docs/rollout_perf_trace_schema_draft.md`。

## 背景目标

我们希望在 verl 中开发一套 rollout 性能工具，目标覆盖三类能力：

1. 采集：在真实 rollout 运行中记录每条 trajectory、每个 turn、LLM 请求、tool call、
   worker 状态和推理引擎内部状态。
2. 可视化：基于采集结果导出 timeline，优先支持 Perfetto 这类现有 trace viewer。
3. 回放：基于采集结果进行 closed-loop replay，用真实推理引擎重放输入/输出长度，
   tool call 用 sleep 和 mock 输出长度替代，用于评估推理引擎优化效果。

当前功能优先级：

- P0：采集和可视化，必须包含 vLLM 内部信息。
- P1：closed-loop replay。
- P2：离线分析、其他 replay 模式、更完整的多引擎内部采集。

设计约束：

- 代码应尽量降低和 verl 训练主流程的耦合。
- 采集模块应可扩展到 vLLM、SGLang、TRT-LLM 等推理引擎。
- 原始 trace 保存全量信息，Perfetto 等可视化格式通过离线转换导出。
- 默认不采集 prompt/tool 原文内容；需要内容时必须通过配置显式打开，避免隐私和文件体积问题。

## 当前代码调研结论

### 已有能力

verl 当前已经有部分 rollout 相关统计：

- `response_length/*`、`prompt_length/*`、`num_turns/*` 等 step summary 会进入 trainer metrics。
- W&B 通过 `Tracking.log(..., step=global_steps)` 记录，每个 trainer step 记录一次，
  不是每个 turn 记录一次。
- 多轮 tool agent 的最终 `response_mask` 中，LLM 生成 token 标为 1，
  tool response/padding token 标为 0，因此具备区分 LLM decode token 和 tool response token 的基础。
- `AgentLoopMetrics` 目前包含 `generate_sequences`、`tool_calls`、`compute_score`、
  `num_preempted`，可以得到较粗粒度的耗时统计。
- fully async 路径中已有 active task、queue size、generated/dropped/staleness 等队列统计。
- Ray timeline 已有配置入口，但只能反映 Ray 运行时事件，缺少 trajectory/turn/tool/vLLM 语义。

### 未满足需求

当前代码还不能满足 rollout perf tool 的核心需求：

- 没有每条 trajectory 的完整生命周期 trace。
- 没有每个 turn 的 LLM 输入长度、输出长度、耗时、tool response token 长度。
- 没有每次 tool call 的输入长度、输出长度、调用耗时、状态、错误信息。
- 没有将 logical request、engine request、worker、server id 关联起来。
- 没有 vLLM scheduler、batching、KV cache、queue 状态的统一采集。
- 没有 Perfetto timeline 导出。
- 没有 closed-loop replay。现有 `rollout_skip` 是复用已有 rollout 输出，不会驱动真实推理引擎。

## 关键代码入口

### Agent Loop

主要文件：

- `verl/experimental/agent_loop/agent_loop.py`
- `verl/experimental/agent_loop/tool_agent_loop.py`

关键位置：

- `AgentLoopWorker.generate_sequences`：per-sample async task 创建和 worker 级上下文。
- `AgentLoopWorker._run_agent_loop`：单条 trajectory 执行入口。
- `ToolAgentLoop.run`：多轮状态机生命周期。
- `ToolAgentLoop._handle_generating_state`：每个 assistant turn 调用 LLM 的位置。
- `ToolAgentLoop._handle_processing_tools_state`：tool call 并发执行和 tool response token 插入位置。
- `ToolAgentLoop._call_tool`：单次 tool call 的执行入口。

### LLM Server

主要文件：

- `verl/workers/rollout/llm_server.py`

关键位置：

- `GlobalRequestLoadBalancer`：request id 到 server id 的 sticky routing 和 inflight count。
- `LLMServerClient.generate`：logical LLM request 转发到具体 server 的入口。

需要注意：

- Agent loop 中的 trajectory/request id 和 engine request id 不是同一个概念。
- `LLMServerClient.generate` 当前会为每个 engine request 生成新的 uuid。
- trace 中必须显式保存 logical request id 和 engine request id 的映射。

### vLLM Server

主要文件：

- `verl/workers/rollout/vllm_rollout/vllm_async_server.py`

关键位置：

- server 初始化阶段创建 `AsyncLLM`。
- `generate` 方法中构造 `SamplingParams`，调用 `self.engine.generate(...)`，
  并消费 async generator 直到最终 `RequestOutput`。

现有返回信息：

- token ids
- log probs
- stop reason
- `num_preempted`，如果 vLLM 输出对象中存在
- MTP/spec decode 相关 stats，如果启用

缺口：

- 当前没有记录 TTFT、queue time、prefill/decode time、per-iteration batch size、
  KV cache 使用、prefix cache 命中等信息。
- `rollout.disable_log_stats` 默认是 `True`，采集 vLLM 内部 stats 时可能需要显式关闭。

### PPO Metrics / Tracking

主要文件：

- `verl/trainer/ppo/metric_utils.py`
- `verl/trainer/ppo/ray_trainer.py`
- `verl/utils/tracking.py`

结论：

- 当前 trainer metrics 适合保存低频 step summary。
- rollout perf trace 是高频、结构化、大体量数据，不适合直接通过 W&B metrics 存储。
- W&B 可以继续记录聚合 summary 和 trace artifact 路径，原始 trace 应保存到本地或共享存储。

## 方案：P0 采集和可视化

## Current P0 Implementation Snapshot (2026-06-03)

Implemented P0 pieces:

```text
Collection spans: trajectory, llm_turn, tool_turn, tool_call, llm_client_request, vllm_engine_request.
Collection counters: vLLM scheduler state, iteration bucket totals, KV cache usage, logical/allocated KV length avg and p95.
Exporters: full Perfetto JSON trace and focused active_counters Perfetto JSON trace.
Configuration: actor_rollout_ref.rollout.perf_trace.engine_internal.sample_interval_ms, default 500 ms.
```

vLLM internal counter semantics:

```text
Scheduler/iteration/KV counters are emitted only while the local vLLM server process has active rollout requests.
When the final active rollout request finishes, pending buckets are flushed and final zero state counters are emitted.
running_requests/waiting_requests are named requests_running/requests_waiting in both new raw records and focused Perfetto output.
The focused active_counters exporter samples vLLM counters over each server's active vllm_engine_request windows.
Raw JSONL remains the source of truth and keeps host/pid/rank/sample interval metadata.
```

Latest validation:

```text
workspace=exp_rollout_perf_trace_retool_vllm020_smoke
run_id=exp_rollout_perf_active_gated_retool_2n8g_20260603_100ms
slurm_job_id=12457801
job_state=COMPLETED
trace_records=826
perfetto_active_counter_events=9967
final_requests_running=0.0 on all four vLLM server lanes
final_requests_waiting=0.0 on all four vLLM server lanes
wandb_url=https://wandb.ai/czqing422-sjtu/verl-fully-async-smoke/runs/retool-vllm020-2n8g-12457801
```

P1 closed-loop replay and P2 offline analyze/alternate replay modes are not implemented yet.


### 新增模块

建议新增：

- `verl/utils/rollout_perf/config.py`
- `verl/utils/rollout_perf/schema.py`
- `verl/utils/rollout_perf/writer.py`
- `verl/utils/rollout_perf/context.py`
- `verl/utils/rollout_perf/collector.py`
- `verl/utils/rollout_perf/export_perfetto.py`
- `verl/utils/rollout_perf/summary.py`

职责：

- `config.py`：配置 dataclass 和开关解析。
- `schema.py`：trace record schema。
- `writer.py`：低开销 trace writer，支持 per-process JSONL/MsgPack 文件和后台 flush。
- `context.py`：使用 `contextvars` 维护 run/step/worker/trajectory/turn/request 上下文。
- `collector.py`：提供统一 `emit_event`、`span`、`counter` API。
- `export_perfetto.py`：离线导出 Perfetto JSON trace。
- `summary.py`：从 raw trace 汇总 step summary。

### 新增配置

建议在 rollout config 下新增独立配置，例如：

```yaml
actor_rollout_ref:
  rollout:
    perf_trace:
      enable: false
      output_dir: null
      format: jsonl
      capture_content: false
      max_samples_per_step_per_worker: null
      flush_interval_s: 5
      export_perfetto: false
      engine_internal:
        enable: false
        backend: auto
        sample_interval_ms: 100
```

不要复用现有 `rollout.trace`，因为它面向 weave/mlflow/trackio 等外部 tracing backend，
并且会采样和记录内容，不适合作为性能 trace 的主路径。

### 采集点

#### Trajectory / Turn

在 agent loop 中记录：

- trajectory 开始/结束
- prompt token length
- total effective token length
- total generated token length
- total tool response token length
- num turns
- num tool calls
- stop reason
- truncate/abort/error 状态

每个 turn 记录：

- turn index
- input token length
- LLM output token length
- tool response token length
- turn latency
- referenced LLM request ids
- referenced tool call ids

#### LLM Request

在 `LLMServerClient.generate` 和 engine server `generate` 中记录：

- logical request id
- engine request id
- worker id / server id
- prompt token length
- requested max tokens
- actual output token length
- TTFT
- total latency
- stop reason
- preempted count
- routing / inflight count snapshot

#### Tool Call

在 `ToolAgentLoop._call_tool` 中记录：

- tool call id
- tool name
- input token/byte length
- output token/byte length
- latency
- status
- error type/message
- timeout/retry/truncation 信息

默认不保存 tool input/output 原文，只保存长度和 hash。需要内容时由配置显式打开。

#### vLLM Internal

P0 必须包含 vLLM 内部信息。优先路径：

1. 在 verl 的 vLLM server adapter 中采集 vLLM 暴露的 request metrics。
2. 使用 `disable_log_stats=False` 打开 vLLM stats。
3. 如果公开返回对象可提供 metrics，则记录 queue/prefill/decode/TTFT 等字段。
4. 如果公开对象不足，再实现 vLLM 版本适配层，在 verl 侧读取 vLLM engine/scheduler/cache
   对象的状态。

原则：

- 优先不修改 vLLM package 源码。
- 如果必须依赖 vLLM 内部属性，也封装在 `rollout_perf` 的 engine adapter 中，
  避免散落在训练逻辑中。

### Timeline / Perfetto

原始 trace 不直接保存为 Perfetto。P0 中保存全量 raw trace，然后离线导出：

- process：node / worker / engine server / tool worker
- thread：trajectory / LLM request / tool call / scheduler
- slice event：trajectory、turn、LLM request、tool call、prefill、decode、queueing
- counter event：inflight request、queue size、batch size、KV used/free、active tasks

这样既能做可视化，也能保留后续分析和 replay 所需的完整信息。

## 方案：P1 Closed-loop Replay

目标：

- 使用真实推理引擎重放生产 rollout 的长度和时序负载。
- tool call 不真实执行，而是按 trace 中记录的耗时 sleep，并 mock 等长输出。
- 重点测推理引擎优化效果，不关注答案内容和 reward。

输入：

- raw trace
- replay 配置
- engine backend 配置
- synthetic prompt/token 生成策略

行为：

- 根据原始 trajectory/turn 顺序驱动 closed-loop。
- 每个 LLM request 使用记录的 input length 和 output length。
- 每个 tool call 使用记录的 latency 和 output length。
- replay 中继续产生新的 perf trace，便于和原始 trace 对比。

限制：

- exact output length 需要进一步验证不同 engine 的控制方式。
- 如果模型提前 EOS，可能需要使用 logits processor、stop 设置或专用 mock decode 策略。
- open-loop replay 先记录为 TODO，不作为第一版实现目标。

## 方案：P2 分析

P2 分析从 raw trace 离线计算，不阻塞 P0 可视化。

计划输出：

- step summary
- trajectory latency 分布
- turn count / tool call count 分布
- LLM decode token vs tool response token 占比
- tool latency 分布
- engine queue / batch / KV 使用率趋势
- worker idle/busy 比例
- bottleneck diagnosis
- 原始 trace 与 replay trace 对比报告

分析结果可以保存为：

- markdown/html report
- JSON summary
- W&B summary metrics 或 artifact

## 多引擎扩展设计

抽象层建议：

- common collector：只定义统一事件和字段。
- engine adapter：负责把 vLLM/SGLang/TRT-LLM 内部状态转成统一 schema。
- engine request wrapper：在各 engine server 的 `generate` 方法周围统一打点。

第一阶段优先 vLLM，因为当前验证 case 使用 vLLM，并且 P0 要求必须包含 vLLM 内部信息。

后续 SGLang/TRT-LLM：

- 先接 request-level timing 和 token length。
- 再逐步接各自 scheduler/cache/batching 内部状态。

## 开发计划

### Phase 0：准备

- 使用干净分支 `rollout-perf-tool-dev`。
- 保存 schema draft 和当前设计计划。
- 以 ReTool 多节点 vLLM 0.20 case 作为验证样例。

### Phase 1：P0 基础 trace

- 新增 `verl/utils/rollout_perf` 模块。
- 新增配置项。
- 实现 writer/context/collector。
- 接入 agent loop、tool agent loop、LLM server client。
- 产出 raw trace。
- 增加基础单元测试。

### Phase 2：P0 vLLM internal

- 接入 vLLM async server request-level metrics。
- 采集 TTFT、latency、output length、stop reason、preemption。
- 验证 vLLM 0.20 是否能通过公开 metrics 获取 queue/prefill/decode/cache 信息。
- 如公开 metrics 不足，实现 verl 侧 vLLM adapter 内部状态采样。

### Phase 3：P0 可视化

- 实现 raw trace 到 Perfetto JSON 的离线导出。
- timeline 中展示 worker、trajectory、LLM request、tool call、engine scheduler/counter。
- 在验证 case 中产出可打开的 Perfetto 文件。

### Phase 4：P1 Closed-loop Replay

- 实现 replay runner。
- mock tool call sleep 和等长输出。
- 使用真实 engine 重放 recorded input/output lengths。
- 产出 replay trace。
- 和原始 trace 做基础对比。

### Phase 5：P2 分析和扩展

- 离线分析报告。
- W&B summary/artifact 接入。
- SGLang/TRT-LLM adapter。
- open-loop replay 和其他 replay 模式。

## 当前开放问题

- vLLM 0.20 可公开获得哪些 scheduler/KV/batching metrics 需要在容器内实际验证。
- exact output length replay 的控制方式需要针对 vLLM/SGLang/TRT-LLM 分别确认。
- 是否需要采集 prompt/tool 原文内容应保持默认关闭，只在明确需要 debug 时打开。

## P0 Implementation Status - vLLM KV Length Sampling

Validated on DFW with ReTool 2 nodes x 8 H100:

```text
run_id=exp_rollout_perf_kvlen_retool_2n8g_20260603_100ms
slurm_job_id=12456997
state=COMPLETED
verl_commit=86338a0c46968431d872617f038c46ece0103f28
image=verlai/verl:vllm020.dev1
vllm_version=0.20.2
sample_interval_default_ms=500
sample_interval_validation_ms=100
wandb_url=https://wandb.ai/czqing422-sjtu/verl-fully-async-smoke/runs/retool-vllm020-2n8g-12456997
```

Implemented P0 pieces:

```text
raw JSONL rollout trace writer and sampled spans/counters
Perfetto full trace export
Perfetto focused active-counter export
agent_loop_worker active counters
vllm_server active request counters
vLLM StatLogger scheduler/iteration sampled counters
vLLM request logical/allocated KV length avg/p95 sampled counters
```

Sampling semantics:

```text
Default engine_internal.sample_interval_ms is 500.
Small validation runs can override to 100 through Hydra.
Scheduler state counters use latest value per window, except kv_cache_usage_ratio uses max per window.
Iteration counters are bucket sums per window.
Request KV length counters are sampled by a background server-side sampler over active requests.
logical_kv_len = prompt tokens + cumulative decoded tokens.
allocated_est_kv_len = ceil(logical_kv_len / block_size) * block_size.
Perfetto focused counter args contain only value; metadata remains in raw JSONL.
```

Remaining P0/P1 work:

```text
Add equivalent engine adapter contracts for SGLang/TRT-LLM without changing the raw trace schema.
Add richer vLLM per-iteration prefill/decode timeline slices if public hooks expose stable timestamps.
Add step_summary.jsonl and low-cardinality W&B artifact/path logging.
Start P1 closed-loop replay after P0 visualization is stable.
```

Detailed runbook:

```text
/home/scratch.ziqingc_gpu/06_codex_projects/06_verl/00_basic/exp_rollout_perf_trace_retool_vllm020_smoke/RUNBOOK.md
```
