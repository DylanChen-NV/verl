# Dynamic Recompute Trajectory 监控与 Mooncake-first 后续计划

> 文件名为兼容历史 handoff 和引用继续保留 `flexkv_followup`；本文内容已由 Mooncake-first 主线取代原 FlexKV-first 执行计划。

## 0. 当前方向与进度

架构和实验主线以 `dynamic_kv_backend_compat_design.md` 为准：先实现 Mooncake 的 sleep-enabled abort KV offload/reuse，完成 A/B/C 与 trajectory/收益分析；FlexKV 保留现有诊断成果，正式 sleep 支持列入后续 TODO。

历史 FlexKV 实现只作为已完成实验的证据与问题定位参考，不作为当前接口设计来源。若历史补丁、脚本或时序与 `dynamic_kv_backend_compat_design.md` 冲突，一律以当前设计为准；迁移 FlexKV 时重新按当前公共 barrier、自动 hit/miss 和最小 VERL 改动边界实现。

截至 2026-07-16：

- recompute 测量和 8K/32 场景筛选已完成。
- A64 预检无有效 retry；A80 预检通过，`rollout.total_rollout_steps=80` 已固定。
- 历史正式 A80（Slurm `14046372`）已完成，MLflow、strict report 和 static Grafana 验收通过；cycle-1 queue-excluded recompute union 为 `1122.457 ms`。
- FlexKV r40 已证明 no-sleep 条件下 abort complete-block PUT、delayed-free 和 visibility barrier；未证明 GET/reuse，正式 sleep 路径受 CUDA VMM registration 阻塞。
- Mooncake 历史实验已严格证明动态新增 replica 的跨节点 reuse，但尚未实现和验证“abort final-save -> barrier -> sleep -> retry GET”的完整目标链路。
- 当前没有运行中的 DFW 作业或固定调试资源。

最终 A/B/C 必须在相同 Mooncake 代码、镜像、资源和 `total_rollout_steps=80` 下运行。历史 A80 作为监控和场景基线保留；若最终 Mooncake 补丁或镜像与历史 A 不同，则重新运行 feature-off 的 A，不能直接混用旧 A 做最终净收益比较。

## 1. 固定实验样本

取消继续扩展 recompute 占比实验，不再追求多 cycle 平均，也不调整 dynamic policy 强制每 step 切换。

后续统一使用已有最大占比场景的第一个 dynamic cycle 作为功能开发和 A/B 样本：

```text
历史任务：Slurm 13662884
资源：DFW，2 节点 x 8 H100
模型：Qwen3-30B-A3B BF16
rollout：vLLM TP2
trainer：Megatron TP2/PP2/EP4
response limit：8K
CONCURRENT_SAMPLES_PER_REPLICA：32
ROLLOUT_MAX_NUM_SEQS：32
required_samples：16
staleness_threshold：3
dynamic_scaling_deactivate_ratio：0.25
数据：DAPO-Math-17k
目标样本：step 0 的 dynamic_cycle_id=1
```

该 cycle 的历史测量结果：


| 指标                             | 结果       |
| ------------------------------ | -------- |
| completed retries              | 144      |
| reaborted attempts             | 206      |
| queue-excluded recompute union | 6.895 s  |
| 平均 global parameter-sync 周期    | 37.983 s |
| 单次 dynamic step 占比             | 18.15%   |
| 全运行摊销占比                        | 2.59%    |


这里的 `18.15%` 只作为该受控 cycle 的历史基线，不再解释为稳定平均收益或默认 recipe 收益。后续只要求复现实验时确实出现一次同类型的 activate → deactivate → abort → retry/recompute 链路，并保持主要配置一致。

本计划只分析 step 0 的首个 dynamic cycle；后续少量 step 只用于让现有 fully-async 流程自然结束，不进入统计和可视化，所有结果均按 `dynamic_cycle_id=1` 过滤。

为节省时间，本轮不修改退出逻辑，不增加单步停止参数。只通过减小 `rollout.total_rollout_steps` 缩短任务，同时保持 8K、`CONCURRENT_SAMPLES_PER_REPLICA=32`、模型、资源和 dynamic 配置不变。当前近似关系为：

```text
total_train_steps
  = floor(rollout.total_rollout_steps
          / (required_samples × trigger_parameter_sync_step))
```

当前 `required_samples=16`、`trigger_parameter_sync_step=1`。以下是已经执行过的预检候选：

| `rollout.total_rollout_steps` | 预计 parameter-sync steps | 用途 |
|---:|---:|---|
| 64 | 4 | 已预检：无 completed retry，未通过 |
| 80 | 5 | 已预检通过，A/B/C 固定使用 |
| 96 | 6 | 未执行，不再需要 |

预检已完成：A64（Slurm `14042455`）没有 completed retry，A80（Slurm `14043414`）得到 64 个 retry attempts 并通过。后续 A/B/C 固定 `rollout.total_rollout_steps=80`；历史正式 A80（Slurm `14046372`）用于监控基线，最终比较按相同 Mooncake 代码和镜像决定是否重跑 A。

## 2. 结合 Grafana trajectory 监控观测 recompute

### 2.1 目标与数据流

参考 `/home/scratch.ziqingc_gpu/06_codex_projects/06_verl/07_monitoring` 已跑通的 MLflow + static Grafana v2 方案，把 8K/32 首个 dynamic cycle 接入：

```text
VeRL rollout trace
→ MLflow trace 持久化
→ MLflow public API dump
→ strict recompute / KV offload event join
→ static Grafana trajectory dashboard
```

Grafana 是唯一的 trajectory 可视化与最终报告入口；MLflow dump、strict report 和 joined CSV/JSON 是可复算的数据依据。

最终需要针对一条 logical trajectory 看清：

```text
初次 generate
→ hybrid replica 执行
→ dynamic deactivate
→ request abort
→ partial token 状态
→ abort KV put 与 barrier
→ retry submit
→ standalone replica 重新调度
→ Mooncake get / recompute prefill
→ first new output
→ trajectory 完成或再次 abort
```

### 2.2 可直接复用的能力

`07_monitoring` 已验证：

- VeRL 原生 `actor_rollout_ref.rollout.trace.backend=mlflow`。
- MLflow SQLite 持久化和 trace UI。
- `MlflowClient.search_traces` 导出 lossless `traces.jsonl`、`spans.csv`、summary 和 manifest。
- 为 `LLMServerClient.generate` 添加 `verl.rollout.server_id`，区分 serving replica。
- static Grafana 13 dashboard、provisioning、启动和服务验收流程。
- Grafana trajectory timeline、per-replica active generate、replica selector 和 shared crosshair。
- Grafana 静态方案不依赖 Prometheus、Tempo 或自定义 datasource。

直接复用或适配：

```text
scripts/dump_mlflow_traces_with_replicas.py
scripts/export_mlflow_trace_to_grafana_v2.py
scripts/validate_mlflow_grafana_v2.py
scripts/start_mlflow_static_grafana_v2.sh
scripts/validate_grafana_service_v2.py
```

现有 Grafana v2 exporter 的输入层需要适配为直接读取 MLflow dump 和 joined recompute/KV-offload 表，不依赖其他可视化格式的中间 trace。

当前 fully-async 代码已经具备基础 trace 点：

```text
AgentLoopWorker._run_agent_loop()
└─ rollout_trace_attr(...)             # trajectory 根上下文

FullyAsyncLLMServerClient.generate()
└─ @rollout_trace_op                   # 覆盖完整 partial-retry 循环

LLMServerClient.generate()
└─ @rollout_trace_op                   # 每次实际 server generate attempt
```

因此不需要移植历史 PR #6680 的 tool-call 专用补丁。只需验证 fully-async 路径确实落 trace，并补充 recompute、put 和 barrier 的关联信息。

### 2.3 需要补充的关联字段

strict recompute/KV offload 事件至少包含：

```text
logical_request_id
attempt_id
dynamic_cycle_id
engine_request_id
replica_id
node_id
T_submit / T_sched / T_first
T_put_enqueue / T_put_launch / T_put_done / T_put_visible
T_retry_gate_enter / T_retry_gate_release
initial_computed_tokens
retry_prefix_tokens
put_status
status=completed|reaborted|incomplete|invalid
```

MLflow generate span 至少补充：

```text
verl.recompute.logical_request_id
verl.recompute.attempt_id
verl.recompute.dynamic_cycle_id
verl.recompute.is_retry
verl.recompute.partial_tokens
verl.recompute.policy_version
verl.rollout.server_id
verl.rollout.replica_id
verl.kv_offload.put_status
```

关联主键：

```text
(logical_request_id, attempt_id, dynamic_cycle_id)
```

不能只靠 span 顺序推断 retry，也不能只靠随机化后的 vLLM internal request ID。保留 `engine_request_id` 用于 scheduler event join，但 logical ID 才是 trajectory 生命周期主键。

### 2.4 最小代码和脚本改动

改动限制在隔离的 monitoring-enabled 实验副本中，不覆盖当前 recompute 基线。

1. 在 fully-async retry loop 中维护 `attempt_id`，对每次 inner `LLMServerClient.generate` span 写入 logical request、attempt、cycle、partial token 和 policy version。
2. 复用最小 server attribution 逻辑，在选中 server 后写入 `verl.rollout.server_id`。
3. 确认每个 trajectory 完成以及 rollouter 退出前执行 MLflow async trace flush。
4. 保留 strict `SUBMIT/SCHEDULED/FIRST_OUTPUT/REABORT` 和 KV put/barrier 打点，不用 MLflow span 替代 queue-excluded recompute 测量。
5. 新增 join 工具，按关联主键生成 `trajectory_recompute_attempts.csv` 和 summary。
6. 适配 `export_mlflow_trace_to_grafana_v2.py`，直接读取 `spans.csv` 与 joined attempts，输出自包含 Grafana dashboard 和静态数据文件。
7. 扩展 Grafana 验收脚本，校验 dynamic、abort、put、barrier、retry、get、recompute 和 replica 数据帧。

### 2.5 运行配置

以历史 8K/32 配置为基础，只增加监控项：

```text
actor_rollout_ref.rollout.trace.backend=mlflow
actor_rollout_ref.rollout.trace.token2text=False
MLFLOW_TRACKING_URI=sqlite:///<persistent-run-dir>/mlflow.db
```

`token2text=False`，避免 8K token 内容显著放大 trace。首轮不使用随机 span 抽样；只采集目标首个 dynamic cycle，或运行后只验收 `dynamic_cycle_id=1`。

MLflow project、experiment、DB、artifact、raw log、strict report、joined data 和 Grafana 输出统一放入独立持久化 run directory。

### 2.6 执行步骤

#### A. 兼容性 smoke

使用小模型或缩短输出跑一次 fully-async dynamic cycle，验证：

- trajectory 根 trace 能创建。
- outer generate 和 inner attempt spans 都存在。
- abort 后同一 logical trajectory 下出现多个 attempt spans。
- trace flush 后 MLflow public API 能完整查询。
- strict event 与 span 可以通过关联主键一一 join。
- Grafana exporter 和 dashboard validation 可以从 dump/join 数据直接通过。

smoke 只验证链路，不产出性能结论。

#### B. 8K/32 正式监控实验

复用历史 8K/32 配置，在 DFW 2×8 H100 上运行。固定使用 `rollout.total_rollout_steps=80`；A、B、C 使用同一配置，并只分析其中的首个 dynamic cycle：

```text
CYCLE_START(step=0)
→ hybrid requests aborted
→ put + barrier
→ retry attempts
→ valid completed/reaborted 分类
→ CYCLE_END
```

不要求后续 step 再次 activate/deactivate，也不以复现恰好 `18.15%` 作为通过条件。

#### C. 数据导出、合并与 Grafana 展示

运行结束后生成：

```text
MLflow:
  traces.jsonl
  spans.csv
  trace_acceptance_summary.json
  dump_manifest.json

Recompute/KV offload:
  queue_excluded_report.json
  queue_excluded_report.txt
  put_barrier_events.csv

Joined:
  trajectory_recompute_attempts.csv
  trajectory_recompute_summary.json

Grafana:
  dashboard JSON
  provisioning files
  static CSV/JSON data frames
  grafana_export_acceptance.json
  grafana_service_acceptance.json
  browser acceptance evidence
```

join 后每个 completed attempt 的 strict `[T_sched, T_first]` 应位于对应 generate attempt 的 `[span_start, span_end]` 内；queue 区间 `[T_submit, T_sched]` 单独展示，不能计入 recompute。

Grafana 至少包含：

1. 按 serving replica 分组的 trajectory execution timeline。
2. 首 cycle 的 dynamic/abort/put/barrier/retry/get/recompute 全局时间线。
3. per-replica active generate 曲线和 replica selector。
4. aborted attempt 明细与 put/reuse status 表。
5. recompute union、put batch wall 和 barrier wait 汇总 panel。

### 2.7 验收标准

1. 每个 `RETRY_SUBMIT` 都能关联到唯一 logical trajectory 和 attempt。
2. 每个 completed attempt 都有唯一 `SUBMIT/SCHEDULED/FIRST_OUTPUT`，时间顺序合法。
3. 每个 reaborted attempt 在 Grafana 数据和 strict report 中状态一致，并从 recompute union 排除。
4. 所有 generate attempt 都有 serving server/replica attribution，能区分 hybrid producer 与 standalone retry consumer。
5. outer trajectory 覆盖 abort 前生成、retry 和最终完成，而不是把 retry 记录成独立 trajectory。
6. MLflow dump 中所有目标 spans 都有完整 start/end timestamp。
7. strict recompute union 与 reporter 结果一致；Grafana 只展示，不重新定义计算口径。
8. Grafana 所有 dashboard targets 返回数据帧且无 query error；replica 筛选、统一绝对时间窗和 shared crosshair 生效。
9. 真实浏览器验收能查看首 cycle 时间线，并定位 `T_put_visible <= T_retry_submit`。
10. 作业退出前 trace flush 完成，持久化 DB、joined data、dashboard JSON 和 artifacts 可以离线重建。
11. 监控开启后 dynamic、abort、retry 和训练流程正常完成；telemetry failure 不得改变业务控制流。

### 2.8 完成产物

- 一份可复现 runner，固定代码版本、镜像、模型、数据和 8K/32 参数。
- 一份监控补丁及其 SHA256，和无监控基线隔离。
- MLflow public-API dump、strict recompute/KV-offload report、join 结果。
- Grafana dashboard JSON、provisioning、静态数据、服务验收 JSON 和浏览器证据。
- 一份可通过 SSH 转发访问的 Grafana dashboard URL 与重启 runbook。
- 一份结论：从单条 logical trajectory 解释 abort 前后执行位置、partial token、put barrier、queue、recompute 和再次 abort 情况。

## 3. Mooncake-first：实现 abort KV offload/reuse

### 3.1 目标与范围

目标闭环：

```text
hybrid request 已生成 partial tokens
-> dynamic deactivate 停止新请求进入 hybrid
-> abort in-flight attempts
-> Mooncake 保存所有完整计算 KV blocks
-> aggregate barrier 等待所有 PUT 终止，成功项可跨节点 lookup
-> retry gate 打开
-> standalone replica retry，lookup hit 时 GET，miss 时完整 recompute
-> hybrid 清理 local KV/weights，但保留 external Mooncake store
-> sleep(level=2)，trainer 使用释放后的 GPU
```

首版约束：

- sleep-enabled dynamic hybrid 是正式目标；non-sleep 只作为数据链路 smoke。
- 使用 cycle-level aggregate barrier，不返回逐请求 `reusable/fallback`。
- consumer connector 自行通过 hit/miss 选择 GET 或 recompute。
- 外部 store 同时只服务一个 policy version；新 policy 生效前完成旧 retry 并清理旧 KV。
- 不新增 VERL lifecycle class，不实现通用 capability negotiation。
- 不新增 Mooncake worker API/class；复用现有 metadata save、异步 PUT 和 `get_finished()` 完成通知。
- Mooncake 首版不增加 `before_device_sleep()/after_device_wake()`。
- 不测试 SSD，不处理多 policy overlap 和长期容量淘汰。

### 3.2 最小代码改动

#### Mooncake connector

| 动作 | Class / 函数 | 首版行为 |
|---|---|---|
| 扩展 | `MooncakeStoreScheduler.request_finished()` | `FINISHED_ABORTED` 时按 `floor(num_computed_tokens / block_size)` 构造完整 block save spec。 |
| 扩展 final metadata | `MooncakeStoreScheduler.build_connector_meta()` | abort request 进入 finished 后，将 final save metadata 放入 vLLM 已有 finished-only 零 token scheduler step。 |
| 使用现有 worker 数据面 | `MooncakeStoreWorker` 的 metadata save/PUT 路径 | final metadata 到达后沿用普通异步 PUT，不新增 abort 专用 worker 方法。 |
| 使用现有 I/O 与完成通知 | `MooncakeStoreWorker.get_finished()` | 在零 token step 中发起 metadata 对应的 PUT/GET，并轮询此前传输完成状态；由现有 connector completion 链路完成 delayed-free 和 aggregate barrier。 |
| 使用现有实现 | `MooncakeStoreConnector.get_num_new_matched_tokens()`、`MooncakeStoreWorker.lookup()` | retry 自动 lookup；hit 时 GET，miss 时完整 recompute。 |
| 使用现有实现 | `MooncakeStoreConnector.register_kv_caches()` | 复用当前 GPU buffer registration；首版不增加 sleep hook。 |
| 调整调用时机 | `MooncakeStoreConnector.reset_cache()` | sleep 时不调用；单 policy 切换、旧 retry 完成后再清理外部 KV。 |

#### 公共路径

| 动作 | Class / 函数 | 首版行为 |
|---|---|---|
| 扩展 | `HybridCheckpointManager.abort_replicas(checkpoint_kv=True)` | 把开关传入 vLLM，并等待 aggregate barrier；不返回逐请求 hit/miss。 |
| 使用现有参数 | `AsyncLLM.pause_generation(mode="abort", clear_cache=False)` | abort 时不提前 reset connector，使 Mooncake 能完成 PUT。 |
| 使用现有 barrier 编排 | `EngineCore.pause_scheduler()` / connector completion 路径 | 复用 vLLM finished-only step 和 `get_finished()`，直到 delayed-free request 完成后 `pause_generation()` 才返回。 |
| 扩展接线 | `EngineCore.sleep(..., reset_connector=False)` | 调用已有 `_reset_caches(reset_connector=False)`，清理 local prefix/KV 后执行 level-2 sleep，但保留 external store。 |

优先复用现有 `RequestTracker`、connector metadata、finished-only 零 token step、`get_finished()` 和 delayed-free 协议，不新增 Mooncake worker API/class，也不新增 EngineCore dispatch 接口。`pause_generation()` 自身作为 barrier；VERL 只在调用前后记录等待区间。

当前 worker patch 中，strict SEND/RECV trace 仅用于验收观测；TCP CPU staging 仅是固定 DFW TCP 环境的传输兼容补丁。二者都不是 abort checkpoint 的控制语义，也不作为新增 worker 接口。

### 3.3 强制时序与 barrier

```text
DynamicResourceController.deactivate_hybrid_replicas()
-> close retry gate
-> remove hybrid replicas from router
-> await HybridCheckpointManager.abort_replicas(checkpoint_kv=True)
   -> pause_generation(mode="abort", clear_cache=False)
   -> MooncakeStoreScheduler.request_finished(FINISHED_ABORTED)
   -> delayed-free source blocks
   -> existing finished-only zero-token step dispatches final metadata
   -> existing MooncakeStoreWorker.get_finished() launches PUT and polls prior completion
   -> all PUT terminal
   -> successful PUT globally lookup-visible
   -> no connector source-GPU read
   -> aggregate barrier returns
-> open retry gate
-> retry logical requests on standalone replicas
-> sleep_replicas(reset_connector=False)
-> trainer starts after hybrid GPU memory is released
```

硬约束：

```text
T_retry_submit >= T_put_visible_barrier
T_sleep_start >= T_no_source_gpu_read
T_trainer_start >= T_sleep_done
```

retry 不等待 sleep 完成；retry 路径可以与 source sleep 和随后的 trainer 计算重叠。PUT 失败或超时必须先进入安全终态，consumer 随后 lookup miss 并完整 recompute，不能永久阻塞 trainer。

### 3.4 Sleep 与单 policy 生命周期

正式路径：

```text
abort barrier complete
-> _reset_caches(reset_connector=False)
-> EngineCore/CuMemAllocator.sleep(level=2)
-> external Mooncake KV remains available
-> standalone retry lookup/GET
```

参数同步进入新 policy 前：

```text
old-policy retries terminal
-> MooncakeStoreConnector.reset_cache()
-> external old KV removed
-> new policy becomes routable
```

如果现有 fully-async 流程无法保证旧 retry 在新 policy 请求进入前完成，则立即停止首版实现并补 `policy_version` namespace，不能复用跨权重 KV。

### 3.5 A/B/C 实验定义

A/B/C 使用同一最终代码、镜像、DFW `2×8 H100`、Qwen3-30B-A3B BF16、8K/32、DAPO-Math-17k 和 `rollout.total_rollout_steps=80`：

| 组别 | abort PUT | barrier | consumer external lookup | 目的 |
|---|---:|---:|---:|---|
| A | OFF | OFF | OFF | recompute baseline |
| B | ON | ON | 强制 miss/OFF | 测 PUT、可见性和 barrier 开销 |
| C | ON | ON | ON | 测 GET/reuse、recompute reduction 和净收益 |

实验开关放在隔离 runner/config 中，不污染默认 recipe。B 必须保留 PUT 和 barrier，但让 consumer lookup 返回 miss；不能通过关闭 Mooncake connector 来伪造 B。

比较口径：

- `B - A`：PUT、visibility 和 barrier 的端到端开销。
- `C - B`：GET/load 开销与 reuse 减少的 recompute。
- `C - A`：最终净收益。

### 3.6 监控字段与分析指标

每个 attempt 至少记录：

```text
logical_request_id, attempt_id, dynamic_cycle_id
producer_replica/node, consumer_replica/node
T_abort_emit
T_put_enqueue, T_put_launch, T_put_done, T_put_visible
T_retry_gate_enter, T_retry_gate_release
T_retry_submit, T_retry_scheduled
T_get_start, T_get_done, T_first_token
partial_tokens, blocks_put, tokens_put, bytes_put
put_status, lookup_hit_tokens, retry_status
```

cycle 级指标：

```text
put_batch_wall = T_put_visible_barrier - min(T_put_launch)
deactivate_increment = T_deactivate(B) - T_deactivate(A)
recompute_saved = T_recompute_union(B) - T_recompute_union(C)
net_cycle_gain = recompute_saved - deactivate_increment - get_load_overhead
net_cycle_gain_ratio = net_cycle_gain / T_dynamic_cycle(A)
```

继续使用 queue-excluded `[T_sched, T_first]` 区间并集，排除 queue 和 re-aborted attempt。不能把并发请求时间或 barrier wait 直接求和当作 cycle 墙钟。

### 3.7 执行步骤

1. 保留已完成的 Mooncake 隔离代码、runner 和旧 A/B/C 产物；旧 B/C 仅作为严格链路未闭环的诊断证据，不进入收益结论。
2. 验证 abort final metadata dispatch：`_finished_save_metas` 必须进入 vLLM 现有 finished-only step，不新增 worker API/class 或 EngineCore dispatch 接口。
3. 使用真实 connector barrier：在 `pause_generation()` 调用前后记录等待区间，并通过 `FINAL_SAVE_DISPATCHED`、`FINISHED_SEND` 和返回顺序证明现有 delayed-free completion 生效。
4. 补 integration test：abort 后不再发生普通 scheduler step，仍必须观察 `FINAL_SAVE_QUEUED -> SEND_DONE/FAILED -> barrier`；同时覆盖 block 取整、零完整 block、failure/timeout 和 `reset_connector=False`。
5. 在固定 DFW `2×8 H100` persistent debug 资源上跑单 request non-sleep smoke，验证原 aborted request 自身完成 PUT，retry 可 forced-miss 或正常 hit。
6. 跑 sleep-enabled strict cross-node smoke，验证 PUT terminal 后才 retry/sleep，external KV 跨 sleep 保留，另一节点 retry 命中同 logical request 的大段 partial prefix。
7. 使用最终相同代码和镜像按 A、B、C 各运行一次正式 8K/32/A80；运行期间持续展示进度。B 必须有原 request SEND 但 consumer forced miss，C 必须有同 request SEND/RECV。
8. 导出 MLflow、strict report、joined CSV/JSON 和 Grafana；按相同 target cohort 完成 A/B/C 分析。若 Top 净收益结果波动明显，再根据总测试时长决定是否复测。
9. 实验完成后固化代码 commit、patch、runner、镜像、配置、日志、SHA256 和重跑命令。

固定调试资源用于实现和 smoke；正式 A/B/C 在同一资源形态和镜像下运行，避免频繁排队和镜像冷启动。

### 3.8 验收条件

功能验收：

1. abort 只保存完整计算 blocks。
2. aggregate barrier 返回前，所有 PUT 已终止，成功项可 lookup，connector 不再读取 source GPU。
3. 每个 retry 满足 `T_retry_submit >= T_put_visible_barrier`。
4. sleep 清理 local KV/weights，但不删除 external Mooncake KV。
5. trainer 在 sleep 后正常执行，无 rollout 显存残留导致的 OOM。
6. C 组存在同一 logical request 的跨节点 strict proof；producer SEND 必须来自原 aborted attempt，且 `hit_tokens / complete_block_prefix_tokens >= 95%`，不能用 sibling prompt 的短 KV 命中代替。
7. PUT 失败项通过 lookup miss 自动 recompute，step 能正常结束。
8. 新 policy 生效前旧 external KV 已清理。

报告验收：

1. A/B/C 三组完整 Grafana trajectory。
2. 逐 attempt 时间线和跨节点 provenance。
3. recompute tokens/time 分布及 queue-excluded union。
4. put batch wall、有效带宽、visibility lag、barrier wait。
5. Mooncake hit tokens/hit ratio 和 strict reuse 证据。
6. `B-A`、`C-B`、`C-A` 同表比较及 net cycle gain。

### 3.9 最终产物

```text
code:
  isolated VERL/vLLM/Mooncake patch and commits
  A/B/C runner and config

evidence per run:
  raw log, config snapshot, environment manifest
  MLflow traces.jsonl and spans.csv
  queue_excluded_report.json/txt
  put_barrier_events.csv
  trajectory_recompute_attempts.csv
  strict_cross_node_reuse_report
  Grafana dashboard/provisioning/static data/acceptance

summary:
  A/B/C comparison table
  recompute reduction
  put and get overhead
  net cycle gain and ratio
  reproduction README and SHA256
```

## 4. FlexKV 后续 TODO

### 4.1 已完成且保留的证据

本节中的旧 patch/commit 不直接进入 Mooncake 主线，也不保证未来原样复用。它们只证明历史运行达到的行为边界；后续 FlexKV 代码必须重新对齐当前设计。

- GCP B200 strict test 已证明动态新增 hybrid replica 能跨节点 reuse FlexKV KV。
- DFW r40 no-sleep diagnostic 已证明 dynamic abort complete-block PUT、delayed-free、visibility barrier 和 barrier 后 retry。
- r40 未证明 GET/reuse，也未通过 trainer；no-sleep trainer OOM 是预期限制。

### 4.2 当前正式 blocker

vLLM sleep 使用 `CuMemAllocator` CUDA VMM；已测试 FlexKV 路径在 `FlexKVConnectorV1.register_kv_caches()` 阶段因 `cudaIpcGetMemHandle error code 1` 失败。

后续需独立完成：

1. 支持 vLLM sleep allocator 的 GPU registration，或改为不向外部进程导出该 VMM allocation。
2. 若 registration 不能跨 sleep 保持有效，在 `FlexKVConnectorV1` 内增加 sleep 前 drain/deregister 和 wake 后 re-register。
3. 实现/确认 GMS/index 的全局可见性 acknowledgement。
4. 将现有 abort PUT patch 对齐现代 `FlexKVConnectorV1.request_finished()/get_finished()`。
5. 复用 Mooncake 已稳定的公共 `abort_replicas(checkpoint_kv=True)`、barrier 和 `reset_connector=False` 路径。
6. 先跑 non-sleep GET/reuse strict test，再跑正式 sleep-enabled B/C。

这些工作不阻塞 Mooncake A/B/C，不在本轮首版实现中修改。

## 5. 当前执行 TODO

按优先级执行：

1. **已完成**：创建 Mooncake-first 隔离代码和 runner，完成首轮 A/B/C，并定位严格证据缺口。
2. **已完成**：验证 final metadata 进入现有 finished-only step；未新增 Mooncake worker API/class 或 EngineCore dispatch 接口。
3. **已完成**：以现有 `pause_generation()` delayed-free completion 作为 aggregate barrier，并通过 5 个容器测试覆盖 final-only dispatch。
4. **已完成**：申请并持续保留 DFW `2×8 H100` persistent debug 资源；会话为 `dyn-moon-final-dispatch-20260717`。
5. **已完成**：`mcpre1` 完成 sleep-enabled strict-cross-node smoke；原请求在 `10.65.4.75` SEND 8,288 tokens，retry 在 `10.65.4.69` RECV 8,288 tokens，首次仅计算 1 token。完整记录见 `next_tests/dynamic_recompute_trajectory_mooncake/MOONCAKE_CROSS_NODE_PRECHECK_20260717.md`。
6. **已完成**：正式 A `ma80r1`、forced-miss B `mb80r1` 和 real-hit C `mc80r2` 均完成；C 严格证明 26 个请求从 `10.65.4.75` 跨节点复用到 `10.65.4.69`，平均 hit ratio 为 99.895%，无 transfer failure。首次 C `mc80r1` 因 barrier target request 为 0 仅保留作诊断，不进入正式结论。
7. **已完成本轮验收**：queue-excluded、strict provenance、target-cohort A/B/C、joined trajectory、三组 Grafana static acceptance 和 C 组 Grafana 13 HTTP service acceptance 均已生成。正式结果见 `next_tests/dynamic_recompute_trajectory_mooncake/MOONCAKE_FORMAL_ABC_RESULTS_20260717.md`；浏览器截图可选，性能主 TODO 为 transfer 并发优化后复测净收益。
8. Mooncake 主线完成后，再启动 FlexKV TODO。
