# Dynamic Hybrid Abort-KV Reuse：Mooncake、FlexKV 与 SGLang 兼容设计

日期：2026-07-16

状态：讨论稿。本文件明确区分上游代码事实、本地实验事实、推论和待实现设计。

## 1. 目标与语义边界

目标不是一般性的 prefix cache 加速，而是处理下面这个 dynamic resource 切换过程：

```text
hybrid rollout replica 正在处理请求
-> VERL 决定将该 replica 归还给 trainer
-> router 停止向该 replica 分发新请求
-> abort 该 replica 上的 in-flight 请求
-> 保存这些 aborted attempt 已经计算完成的全部完整 KV block
-> 等待这些 KV 能够被其他 replica/节点发现和读取
-> 在 standalone rollout replica 上重试对应的 logical request
-> 释放 hybrid rollout 的 GPU 显存并运行 trainer
-> retry 加载已保存 KV，避免重复 prefill/recompute
```

必须满足以下安全约束：

1. 对每个 aborted retry，必须满足 `T_retry_submit >= T_kv_reusable`。
2. 后端仍可能读取 GPU block 时，不能释放 block，也不能让 engine sleep。
3. 只保存已经完整计算的 block：
  `saved_tokens = floor(num_computed_tokens / block_size) * block_size`。
4. producer 和 consumer 必须使用相同模型、相同 policy version 和兼容的 KV layout。
5. 失败时必须回退到完整 recompute，不能无限阻塞 trainer，也不能读取 stale KV。
6. PUT 完成并不足够；必须证明其他 replica 能发现并成功加载该条目。

首版不包括：

- 与 hybrid deactivation 无关的普通 partial rollout KV reuse；
- vLLM 产生 KV、SGLang 消费 KV 的跨引擎互通；
- 跨 policy weight update 复用 KV；
- SSD 测试。

## 2. 当前 VERL 与 vLLM 调用栈

当前隔离分支的 deactivation 顺序是：

```text
[VERL control plane]
DynamicResourceController.deactivate_hybrid_replicas()
-> 可选：关闭本周期 retry gate
-> rollouter.remove_replicas(hybrid_ids)
-> HybridCheckpointManager.abort_replicas()
   -> vLLMReplica.abort_all_requests()
      -> server.abort_all_requests()

         [从这里进入 vLLM]
         -> AsyncLLM.pause_generation(mode=abort，或旧版本等价实现)
-> 可选：打开本周期 retry gate
-> HybridCheckpointManager.sleep_replicas()
   -> vLLMReplica.sleep()
      -> server.sleep()

         [从这里进入 vLLM]
         -> AsyncLLM/EngineCore.sleep(level=2)
```

`vLLMReplica` 和 `server` 仍属于 VERL rollout adapter；`AsyncLLM`、`EngineCore`、scheduler、worker 及其 KV connector 属于 vLLM。Mooncake/FlexKV 的 connector scheduler/worker 位于 vLLM 内部，真正的 store/transfer runtime 才进入对应后端。

本地源码：

- `next_tests/dynamic_recompute_trajectory_flexkv/verl-215fa98/verl/experimental/fully_async_policy/dynamic_scaling/dynamic_resource_controller.py`
- `next_tests/dynamic_recompute_trajectory_flexkv/verl-215fa98/verl/checkpoint_engine/base.py`
- `next_tests/dynamic_recompute_trajectory_flexkv/verl-215fa98/verl/workers/rollout/vllm_rollout/vllm_async_server.py`

当前 FlexKV 实验在 abort 前后增加了 cycle gate。r40 的 no-sleep 实验能够工作，是因为本地修改后的 connector 延迟释放 source block，并且 `pause_generation()` 返回前，connector 工作已经进入观测到的终态。该实验只验证了 abort PUT 的 delayed-free/barrier，没有验证 GET/reuse，也没有验证正式 sleep 路径。

### 2.1 vLLM KVConnectorBase_V1 已提供的能力

MooncakeStoreConnector 和 FlexKVConnectorV1 都实现了 `KVConnectorBase_V1` 的主要数据面，但当前主要覆盖普通 prefix-cache 流程，不等于已经满足 dynamic abort-reuse 语义：


| `KVConnectorBase_V1` 能力        | MooncakeStore | FlexKV | 对当前目标的限制                                                         |
| ------------------------------ | ------------- | ------ | ---------------------------------------------------------------- |
| `get_num_new_matched_tokens()` | 支持            | 支持     | 可用于 retry prefix lookup                                          |
| `update_state_after_alloc()`   | 支持            | 支持     | 基础 allocation metadata 能力可用                                      |
| `register_kv_caches()`         | 支持            | 支持     | FlexKV 在已测试 sleep allocator 下注册失败                                |
| `request_finished()`           | 支持            | 支持     | Mooncake 未证明 abort decode final-save；FlexKV 上游拒绝 abnormal finish |
| delayed-free                   | 支持            | 支持     | 普通异步保存可用；abort 是否触发保存仍需扩展                                        |
| worker `get_finished()`        | 支持            | 支持     | 完成含义不必然等于其他节点已可 lookup/GET                                       |
| connector reset/shutdown       | 支持            | 支持     | 需要把本地清理与外部 KV namespace 清理解耦                                     |


其中，`request_finished()` 在 block 真正释放前执行。返回 `True` 后，connector 临时取得这些 block 的所有权，直到 `get_finished()` 报告完成。

来源：[vLLM KVConnectorBase_V1](https://docs.vllm.ai/en/stable/api/vllm/distributed/kv_transfer/kv_connector/v1/base/)。

### 2.2 vLLM 尚未统一的能力

KVConnector API 当前没有定义：

- `before_sleep()` / `after_wake()`；
- 显式的 `drain(request_ids, timeout)`；
- send completion 究竟表示本地复制完成、持久化完成，还是全局可发现；
- 面向 abort 的 aggregate barrier：所有 PUT 终止、成功项全局可见且无 source GPU read；
- policy-version namespace；
- abort 时保存全部已计算 decode block 的统一规则。

当前 vLLM `sleep(level>=1)` 会暂停 scheduler 并清理本地 prefix state，默认 reset 路径还会 reset connector。作为通用默认行为这是安全的，但与“保存 abort KV 供同 policy retry 立即复用”的需求冲突。公共 sleep API 不能独立表达“清理本地 GPU KV，但保留外部 KV namespace”。

来源：

- [vLLM EngineCore sleep 与 pause](https://docs.vllm.ai/en/v0.23.0/api/vllm/v1/engine/core/)
- [vLLM pause_generation API](https://docs.vllm.ai/en/v0.23.0/api/vllm/engine/protocol/)

## 3. 只支持 Mooncake 的最小正确接入

这里的 Mooncake 指 vLLM `MooncakeStoreConnector`，不是仅用于 P/D transfer 的 `MooncakeConnector`，也不是 FlexKV 内部使用的 Mooncake Transfer Engine。

### 3.1 已有能力

`MooncakeStoreConnector` 实现了 `KVConnectorBase_V1`，使用 MooncakeDistributedStore 作为共享 KV pool，支持跨实例 lookup，并报告异步 send/receive 完成。

来源：

- [MooncakeStoreConnector 使用说明](https://docs.vllm.ai/en/stable/features/mooncake_store_connector_usage/)
- [MooncakeStoreConnector 实现](https://docs.vllm.ai/en/latest/api/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/connector/)
- [MooncakeStore worker](https://docs.vllm.ai/en/latest/api/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker/)

本地实验 `exp_gcp_b200_dynamic_mooncake_strict_reuse_20260625_0007` 已证明：动态激活的新 hybrid replica 能在 `free_cache_engine=True` 条件下跨节点复用 KV。实验同时发现，普通 connector reset 会删除共享条目；为完成严格验证，实验使用了保留 connector store 的补丁，并通过 `ACTOR_LR=0` 保持权重不变。

本地证据：

- `next_tests/dynamic_mooncake_strict_reuse/README.md`
- `next_tests/dynamic_mooncake_strict_reuse/NEXT_TEST_NOTES.md`

### 3.2 当前目标仍缺少的能力

上游 MooncakeStore scheduler 主要跟踪和保存 prefill range。其 `request_finished()` 能在已有 save tracker 时延迟释放，但没有明确的 `FINISHED_ABORTED` 分支来快照全部完整 decode block。

因此，之前的跨节点 prefix reuse 成功，不等价于已经证明 abort 时能够保存长 decode 请求的全部 partial KV。

来源：[MooncakeStore scheduler](https://docs.vllm.ai/en/stable/api/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler/)。

### 3.3 Mooncake-only 最小改动

1. 扩展 MooncakeStore request tracking：请求被 abort 时，生成覆盖到 `num_computed_tokens` 的完整 block save spec。
2. 沿用 vLLM delayed-free：connector 报告 send 完成前，source block 不得释放。
3. 让现有 abort 调用等待 aggregate barrier：所有 PUT 成功或失败终止、成功项可 lookup、connector 不再读取 source GPU。
4. deactivation 中区分本地 reset 和外部 store reset：清理本地 GPU/prefix state，但保留外部 Mooncake store 供 retry lookup。
5. 首版维持单 policy invariant：新 policy 生效前完成旧 retry 并清理外部 store；多 policy overlap 时再增加 namespace。
6. barrier 返回后统一释放 retry；consumer lookup hit 时 GET，miss 时自动完整 recompute。

这是当前最短路径：Mooncake 在已验证环境中能够配合 vLLM sleep，并已具备 shared-store lookup/transfer。主要新增工作集中在 abort final-save、aggregate barrier 和本地/外部 reset 分离。

## 4. 只支持 FlexKV 的最小正确接入

### 4.1 已有能力

从 vLLM 0.17.2 起，FlexKV 以 `FlexKVConnectorV1` 形式进入 vLLM，使用相同的 lookup、`request_finished()`、delayed-free 和 `get_finished()` 基础模型。

来源：

- [FlexKV vLLM adapter 指南](https://github.com/taco-project/FlexKV/blob/main/docs/vllm_adapter/README_zh.md)
- [vLLM FlexKVConnectorV1 API](https://docs.vllm.ai/en/v0.23.0/api/vllm/distributed/kv_transfer/kv_connector/v1/flexkv_connector/)

本次调研检查的 FlexKV main commit 为 `3c968fc09c3d3b4f3f46bd0fcc9c4d6d9bb94ca8`。该版本 adapter 的 `request_finished()` 仍会拒绝 abnormal finish，因此 abort PUT 不是上游已有行为。它仍通过 `KVTPClient` / `TensorSharedHandle` 注册 worker GPU tensor，shared handle 使用 PyTorch reduction 或 `cudaIpcGetMemHandle`。

本地隔离补丁已经在 no-sleep 模式证明：dynamic abort 可以 PUT 全部完整计算 block，并通过 delayed-free 等待完成；但没有证明 GET，也没有证明 sleep-enabled 路径。

### 4.2 相比 Mooncake 多出的 blocker

在已测试的 vLLM/FlexKV 镜像中，启用 vLLM sleep 后会使用 `CuMemAllocator` CUDA VMM allocation，FlexKV 在实际请求前就因 `cudaIpcGetMemHandle` 失败。当前检查的 FlexKV main 中也没有找到明确的 sleep/wake lifecycle hook。

这包含两个独立要求：

1. 初次 GPU 注册必须兼容固定版本组合中的 vLLM sleep allocator。
2. sleep 前必须 drain 并使外部 GPU mapping/registration 失效；wake 后必须重新注册当前 allocation，或者证明原 mapping 仍然有效。

第一项是已经观测到的 blocker；第二项是生命周期必需条件，但失败实验尚未执行到这一阶段。

### 4.3 FlexKV-only 最小改动

1. 保留 abort-PUT 扩展，但基于固定的现代 `FlexKVConnectorV1` 实现，而不是继续维护旧实验 runtime copy。
2. 增加可靠的 aggregate barrier 和安全取消语义：所有 PUT 终止后才能 sleep。
3. 定义“全局可复用”确认，不能把 transfer completion 与 GMS/分布式索引传播完成混为一谈。
4. 支持 sleep allocator 的 GPU 注册。可选方案：
  - 支持相关 CUDA VMM allocation 的 export/import；或
  - GPU copy 留在所属 vLLM worker 内，只向外部 FlexKV worker 暴露 host/stored data。
5. 后续按需增加 device lifecycle hook：停止新 GPU 操作、drain、sleep 前 deregister、wake 后重新注册。
6. 首版沿用单 policy invariant；未来允许多 policy overlap 时再增加模型、policy version 和 KV layout namespace。

所以，除非新的固定 FlexKV/vLLM 版本能单独通过 sleep registration 微测试，否则 FlexKV-only 不是最短的第一阶段方案。“connector 已进入 vLLM”只说明 API/package 集成，不证明 sleep 生命周期兼容。

## 5. 两种方案对比


| 要求 | vLLM + MooncakeStore | vLLM + FlexKV |
|---|---|---|
| 标准 connector lookup/load | 已有 | 已有 |
| delayed-free 基础能力 | 已有 | 已有 |
| 本地跨节点 reuse 证明 | 已完成 sleep-enabled 受控测试 | 已完成 sleep-disabled 测试 |
| abort 时保存全部 decode KV | 尚未证明，需要扩展 | 上游没有，本地已有补丁 |
| aggregate abort barrier | 需要补充 | 上游没有，本地只有观测逻辑 |
| 成功 PUT 的全局可见性 | 需要确认或扩展 | 需要 GMS/index 可见性 API |
| 清理 local KV、保留 external KV | `_reset_caches(reset_connector=False)` 可用，需接入 sleep | 需要设计 |
| vLLM sleep 注册 | 本地 Mooncake 实验已工作 | 已测试镜像失败 |
| sleep/wake device hook | 首版不需要 | FlexKV 后续可能需要 |
| 第一阶段接入成本 | 较低 | 较高，除非新版已解决 sleep |


## 6. 同时兼容 Mooncake 和 FlexKV 的设计

设计应在存储实现之上统一语义，不能让 `DynamicResourceController` 直接调用 Mooncake/FlexKV 私有 API。

### 6.1 VERL 现有流程的最小扩展

VERL 已经具备 router、cycle gate、`abort_replicas()`、`sleep_replicas()`、trainer 和 wake/activate 流程。它们不因 Mooncake/FlexKV 而产生，不增加新的生命周期抽象。

首版只给现有 abort 接口增加一个开关：

```python
async def abort_replicas(
    checkpoint_kv: bool = False,
) -> None: ...
```

当 `checkpoint_kv=True` 时，该异步调用本身就是 barrier，返回必须同时表示：

- 所有相关 PUT 已成功或失败并进入终态；
- 成功 PUT 的 KV 已达到 consumer 可 lookup 的可见状态；
- connector 已停止读取 source GPU，delayed-free 可以结束。

失败的 PUT 不单独返回给 VERL；后续 consumer lookup miss 时自然执行完整 recompute。因此不新增逐请求 `AbortKVResult`。

controller 只做局部调整：

```text
关闭 retry gate
-> 从 router 移除 hybrid replica
-> await abort_replicas(checkpoint_kv=True)
-> barrier 返回后打开 retry gate
-> 调用现有 sleep_replicas()
```

VERL 不直接依赖 Mooncake/FlexKV API，也不判断 retry 最终是 KV hit 还是 miss。

### 6.2 vLLM connector 的首版改动

vLLM 继续使用 `KVConnectorBase_V1` 的 `request_finished()/get_finished()` 和 delayed-free，不增加通用 capability 接口：

- aborted request 进入 `request_finished()` 时，connector 构造截至 `num_computed_tokens` 的 full-block save spec；
- `request_finished()` 返回 delayed-free，worker 异步执行 PUT；
- vLLM abort 路径等待所有 PUT 成功或失败终止，并确认成功项已可被 consumer lookup；
- barrier 完成后，现有 `abort_replicas()` 返回；consumer 后续通过 connector lookup hit/miss 自动选择 GET 或 recompute。

vLLM sleep 还需要把“清理本地 KV”和“reset 外部 connector”分开。内部 `_reset_caches(reset_connector=False)` 已经存在；首版新增的是让公共 sleep 路径能够选择该参数：

```text
sleep(level=2, mode="abort", reset_connector=False)
```

首版 Mooncake 路径不实现 `before_device_sleep()`：前面的 abort barrier 已保证 GPU I/O 结束，并且现有实验中 Mooncake registration 可以跨 sleep 使用。保留一条代码注释，说明未来 FlexKV 如果需要 drain、注销和重建 GPU mapping，可在 vLLM connector 内增加该 hook；VERL 不感知它。

### 6.3 vLLM 内部 backend adapter

Mooncake adapter：

- 构造 aborted request 的最终 save spec；
- 使用已有 delayed-free/send completion；
- 将 store completion 映射为全局可见性；
- 首版只保留当前单 policy 所需的外部 store。

FlexKV adapter：

- 构造同样的 full-block abort PUT；
- 将 FlexKV/GMS 完成映射为全局可见性；
- 实现 sleep-compatible device registration；
- 首版只保留当前单 policy 所需的外部 index 和 KV。

配置选择 backend，但 controller 不按 backend 名称分支：

```yaml
rollout:
  abort_kv_reuse:
    enabled: true
    timeout_s: 30
    require_global_visibility: true
    fallback: recompute
  kv_backend: mooncake_store  # 或 flexkv
```

### 6.4 首版配置与初始化校验

首版固定使用同构节点和固定版本的 Mooncake，不实现通用 capability negotiation。启动时只检查：connector 配置已启用、初始化成功、所有相关 replica 使用相同 backend 配置；失败则 fail-fast 或关闭 abort KV reuse。

未来同时支持 FlexKV、异构版本或动态加入未知节点时，再考虑报告 `supports_device_sleep`、`supports_global_visibility_ack` 等 capability。该扩展不进入 Mooncake 首版改动。

### 6.5 增加 KV cache offload 前后的调用栈差异

下面以 sleep-enabled dynamic hybrid 为主场景。`[不变]` 表示复用现有流程，`[修改]` 表示扩展现有接口，`[启用已有]` 表示 vLLM/Mooncake 已有能力只需配置启用，`[新增]` 表示目标场景确实需要补充的逻辑。

#### 6.5.1 增加前：现有 abort + recompute

```text
[不变][VERL DynamicResourceController]
DynamicResourceController.deactivate_hybrid_replicas()
-> 关闭 retry gate
-> rollouter.remove_replicas(hybrid_ids)
-> HybridCheckpointManager.abort_replicas()
   -> vLLMReplica.abort_all_requests()
   -> VERL vLLM server.abort_all_requests()

      [进入 vLLM]
      -> AsyncLLM.pause_generation(mode="abort")
      -> scheduler 将 in-flight attempt 标记为 FINISHED_ABORTED
      -> 释放其 local KV block

-> 打开 retry gate
-> HybridCheckpointManager.sleep_replicas()
   -> vLLMReplica/server.sleep()

      [进入 vLLM]
      -> EngineCore.sleep(level=2, mode="abort")
      -> 清理 local prefix/KV cache
      -> CuMemAllocator 释放 GPU memory

-> trainer 使用释放后的 hybrid GPU
```

被 abort 的 logical request 在其他 replica 上从 prompt 开始重新 prefill，重算 aborted attempt 已计算过的 token。

#### 6.5.2 增加后：abort checkpoint barrier + 自动 hit/miss

Connector 初始化发生在 vLLM 内部。MooncakeStoreConnector 的创建、`register_kv_caches()` 和 backend buffer 注册是已有能力，首版主要通过配置启用，不属于新增接口：

```text
[不变] vLLM EngineCore / worker 初始化
-> [启用已有] 创建 MooncakeStoreConnector
-> [启用已有] connector.register_kv_caches(kv_caches)
-> [启用已有] Mooncake store.register_buffer(...)
-> [新增，轻量] 启动时校验所有相关 replica 配置一致且初始化成功
```

不增加 vLLM capability 查询和 VERL capability 汇总。deactivation 只扩展现有 abort 调用：

```text
[不变][VERL DynamicResourceController]
DynamicResourceController.deactivate_hybrid_replicas()
-> 关闭 retry gate
-> rollouter.remove_replicas(hybrid_ids)

-> [修改] await HybridCheckpointManager.abort_replicas(
             checkpoint_kv=True,
         )
   -> [修改] vLLMReplica/server 将 checkpoint_kv 传入 vLLM

      [进入 vLLM]
      -> [修改] pause_generation(mode="abort", clear_cache=False)
      -> [不变] scheduler 标记 FINISHED_ABORTED
      -> [新增] connector.request_finished(request, block_ids)
      -> [新增] 根据 num_computed_tokens 构造 full-block save spec
      -> [新增] request_finished() 返回 delay_free=True
      -> [新增] connector 暂时持有 source block
      -> [新增][worker] get_finished() 发起 Mooncake PUT
      -> [新增] 等待所有 PUT 成功或失败进入终态
      -> [新增] 确认成功 PUT 已可被 consumer lookup
      -> [新增] 确认 connector 不再读取 source GPU，结束 delayed-free
      -> [修改] abort 调用作为 barrier 返回，不返回逐请求结果

-> [不变] 打开 retry gate，正常重试所有 logical request

-> [不变] HybridCheckpointManager.sleep_replicas()
   -> vLLMReplica/server.sleep(reset_connector=False)

      [进入 vLLM]
      -> [修改] _reset_caches(reset_connector=False)
                   清理 local prefix/KV，但不清空外部 Mooncake store
      -> [不变] EngineCore/CuMemAllocator sleep(level=2)

      # 未来 FlexKV 如果需要在 sleep 前 drain/deregister GPU mapping，
      # 再在 vLLM connector 内增加 before_device_sleep()；Mooncake 首版不实现。

-> [不变] trainer 使用释放后的 hybrid GPU
```

Barrier 返回后，retry 与 source sleep 可以并行：

```text
路径 1：retry，不等待 source sleep 完成
[不变] router 重试 logical request
-> [启用已有] consumer connector lookup
-> hit: GET 外部 KV，从命中位置继续 decode
-> miss: 从 prompt 开始完整 recompute

路径 2：释放 source GPU
[不变] sleep_replicas()
-> [修改] local reset 不 reset external connector
-> [不变] sleep 完成后 trainer 才能使用该 GPU
```

因此 VERL 不需要知道某个请求是 hit 还是 miss。安全约束只有两个：retry 不能早于 abort barrier 返回；trainer 不能早于 hybrid GPU sleep 完成。

#### 6.5.3 Trainer 后的 wake/activate

Mooncake 首版不增加 wake hook：

```text
[不变] trainer 完成
-> [不变] 现有 wake/activation 流程恢复 hybrid rollout replica
-> [不变] 参数同步
-> [不变] router.add_replicas(hybrid_ids)
```

未来 FlexKV 如果需要重建 GPU registration，再在 vLLM connector 内增加 `after_device_wake()`；VERL 不增加新的 `wake()` 抽象。

#### 6.5.4 单 policy 约束与外部 KV 清理

首版不传 `policy_version`，要求外部 KV store 同时只服务一个 policy 版本。切换到新 policy 前，必须确认旧版本 retry 已完成，并清理旧外部 KV；不能让新旧 policy 的相同 token key 同时参与 lookup。

未来允许多个 policy version overlap 时，再把 `policy_version` 或 weight digest 加入 cache namespace。

### 6.6 Mooncake/FlexKV 角度的改动

#### Mooncake

| 动作 | Class / 接口 | 说明 |
|---|---|---|
| 扩展 | `MooncakeStoreScheduler.request_finished()` | abort 时构造截至 `num_computed_tokens` 的完整 block save spec。 |
| 使用并扩展完成语义 | `MooncakeStoreConnector.request_finished()`、`MooncakeStoreWorker.get_finished()` | 使用 delayed-free；所有 PUT 终止、成功项可 lookup 且无 GPU read 后，aggregate barrier 才完成。 |
| 使用现有实现 | `MooncakeStoreConnector.get_num_new_matched_tokens()`、`MooncakeStoreWorker.lookup()` | consumer 自动 lookup；命中则 GET，miss 则完整 recompute。 |
| 使用现有实现 | `MooncakeStoreConnector.register_kv_caches()` | 注册 GPU KV buffer；Mooncake 首版无需增加 registration hook。 |
| 调整调用时机 | `MooncakeStoreConnector.reset_cache()` | sleep 时不调用；单 policy 切换完成后再清理旧外部 KV。 |

#### FlexKV

| 动作 | Class / 接口 | 说明 |
|---|---|---|
| 扩展 | `FlexKVConnectorV1.request_finished()` | 接受 aborted/abnormal finish，并保存全部完整计算 block。 |
| 扩展 | `FlexKVConnectorV1.get_finished()` | 实现 aggregate barrier，并确认成功 PUT 已在 GMS/index 中全局可见。 |
| 使用现有实现 | `FlexKVConnectorV1.get_num_new_matched_tokens()` | consumer 自动通过 hit/miss 选择 GET 或完整 recompute。 |
| 修复/扩展 | `FlexKVConnectorV1.register_kv_caches()` | 兼容 vLLM sleep allocator，解决 CUDA IPC/VMM registration 失败。 |
| 未来按需新增 | `FlexKVConnectorV1.before_device_sleep()`、`FlexKVConnectorV1.after_device_wake()` | 必要时负责 sleep 前 drain/deregister 和 wake 后重新注册。 |
| 新增 override | `FlexKVConnectorV1.reset_cache()` | policy 切换后清理旧 GMS/index 和外部 KV；sleep 时不调用。 |

#### 公共路径

| 动作 | Class / 接口 | 说明 |
|---|---|---|
| 扩展 | `HybridCheckpointManager.abort_replicas(checkpoint_kv=True)` | 等待 aggregate barrier，不返回逐请求 hit/miss。 |
| 扩展接线 | `EngineCore.sleep(..., reset_connector=False)` | 调用已有 `_reset_caches(reset_connector=False)`，清理 local KV 但保留 external KV。 |

### 6.7 Non-sleep 模式的低成本兼容

Non-sleep 模式复用相同 barrier，只跳过现有 `sleep_replicas()`：

```text
await abort_replicas(checkpoint_kv=True)
-> barrier 返回后正常 retry
-> connector hit 时 GET，miss 时 recompute
-> source replica 保持 awake
```

该模式不需要任何 sleep/wake hook，也不增加第二套状态机。

## 7. Policy version 正确性

相同 token 序列在不同权重下产生的 KV 不同，因此不能让多个 policy version 共用无版本区分的 cache key。

首版暂不传 `policy_version`，而是依赖以下单版本约束：

```text
外部 KV store 同时只服务一个 policy version
-> abort producer 与 retry consumer 使用同一组权重
-> 新 policy 生效前，旧版本 retry 必须全部完成
-> 随后清理外部 KV，再允许新 policy 请求进入
```

如果 fully async 流程未来允许多个 policy version 的请求并存，该约束即不成立；届时必须把 `policy_version` 或 weight digest 加入 cache namespace。

## 8. vLLM 替换为 SGLang 后的兼容性

### 8.1 当前 VERL SGLang 生命周期

当前 VERL 调用：

```text
abort:
  tokenizer_manager.pause_generation(mode="abort")

sleep:
  release_memory_occupation(tags=["kv_cache", "weights"])

wake:
  resume_memory_occupation(...)
  flush_cache()
```

本地源码：

- `next_tests/dynamic_recompute_trajectory_flexkv/verl-215fa98/verl/workers/rollout/sglang_rollout/async_sglang_server.py`

SGLang 不实现 vLLM `KVConnectorBase_V1`。其自然抽象是 HiCache：L1 GPU、L2 host 和可选 L3 storage。Mooncake 是内置 L3 backend，SGLang 也支持动态 storage-backend plugin 配置。

来源：

- [SGLang server arguments](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/server_arguments.md)
- [SGLang Mooncake L3 backend](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/storage/mooncake_store/README.md)

### 8.2 SGLang + Mooncake

Mooncake 是 SGLang 上最简单的存储选择，因为已有 HiCache L3 backend。但当前目标仍需要 SGLang 内部提供 checkpoint 操作：

```text
pause/abort
-> 找到每个 aborted request 对应的完整 KV prefix
-> pin/finalize 对应 radix-cache node
-> 强制或等待备份到 L2/L3
-> 返回逐请求 reusable 状态
-> 释放 L1 GPU memory
```

仅使用公共 `pause_generation` 和 `release_memory_occupation` 无法提供逐请求 barrier。该能力必须在能看到 radix node、block ownership 和 write-back 状态的 SGLang/HiCache 内部实现。

### 8.3 SGLang + FlexKV

当前 FlexKV README 提到支持 SGLang-compatible KV layout，但本次检查的 main tree 中没有维护中的 SGLang adapter 或使用文档；roadmap 仍把 framework integration update 列为后续工作。layout 兼容不等于完整生命周期集成。

两条可能路径：

1. 将 FlexKV 实现为 SGLang dynamic HiCache storage backend，优先放在 L2 之下，由 SGLang 自己管理 GPU 生命周期。
2. 编写直接的 SGLang/FlexKV adapter，参与 radix-cache allocation、eviction、abort finalization 和 memory-saver lifecycle。

路径 1 更适合作为后续方案，因为 SGLang 已定义 storage backend 边界，而且 L2 staging 能隔离 FlexKV 与 GPU sleep allocation。代价是增加一次 host-memory stage，除非后续增加 direct I/O。

### 8.4 跨引擎可以和不可以复用的部分

可以复用：

- VERL cycle gate 和 router ordering；
- 现有 `abort_replicas(checkpoint_kv=True)` 的 aggregate barrier 语义；
- 单 policy 切换约束，以及未来的 namespace 规则；
- consumer connector 自动 hit/GET 或 miss/recompute 的行为；
- telemetry 和严格跨节点验收标准。

必须保留在引擎 adapter 内部：

- vLLM block ID 和 `KVConnectorBase_V1` callback；
- SGLang radix-tree/HiCache node ownership；
- allocator sleep/wake 操作；
- backend-specific global visibility 和 cancellation 语义。

因此不建议在 vLLM 和 SGLang 之间设计万能 connector API。稳定边界只是现有 engine adapter 上 `abort_replicas(checkpoint_kv=True)` 的 barrier 语义；checkpoint 细节由引擎原生实现。

## 9. 推荐实施顺序

1. 固定 VERL、vLLM、Mooncake、FlexKV、SGLang 和 CUDA 的准确版本，不能依据持续变化的 main 分支推断行为。
2. 最小扩展现有 `abort_replicas()`，只加入 `checkpoint_kv` 和 aggregate barrier；无 backend 时保持原有 full recompute。
3. 实现 vLLM + MooncakeStore 的 abort final-save、本地/外部 reset 分离和严格 barrier。
4. 在固定调试资源上验证 `abort -> PUT visible -> sleep -> trainer -> retry GET`。
5. 在设计 allocator 修改前，先运行现代 FlexKV sleep 微测试；失败时通过配置禁用 FlexKV abort reuse。
6. FlexKV 满足相同 barrier 和 sleep 安全条件后接入 vLLM adapter，并复用相同验收测试。
7. 在现有 SGLang rollout adapter 中最小扩展 `abort_replicas()`，由 HiCache 实现 checkpoint barrier，支持 SGLang + Mooncake。
8. 除非上游发布受维护的适配器，否则把 SGLang + FlexKV 作为后续 storage-backend 项目。

## 10. 统一验收测试

每个 backend/engine 组合都必须通过相同的语义测试：

1. 单测：abort 只保存完整计算 block。
2. 单测：所有 PUT 进入终态且成功项可 lookup 前，aggregate barrier 不能返回。
3. 单测：timeout 后停止所有 GPU read，才能 sleep。
4. 单测：新 policy 生效前必须完成旧 retry 并清理外部 KV。
5. 集成：hybrid sleep 后 GPU 显存下降，trainer 不因 rollout 占用而 OOM。
6. 集成：wake 后恢复可工作的本地 KV pool 和 backend access。
7. 严格复用：producer 与 consumer 节点不同、digest/key 匹配、hit tokens 大于零，并且 GET 发生在 PUT 全局可见之后。
8. 收益：相对 PUT-without-GET baseline，排除排队后的 recompute interval 并集下降。
9. 容错：PUT 失败后 consumer lookup miss，并通过完整 recompute 完成 step。
10. 生命周期：至少连续完成三次 activate/deactivate/sleep/wake，无 stale mapping、泄漏任务或 stale-policy hit。
11. Non-sleep smoke test：复用相同 abort PUT/GET barrier，跳过 sleep/wake。

## 11. 当前建议

第一阶段先实现 Mooncake：只给 VERL 现有 `abort_replicas()` 增加 `checkpoint_kv`，由 vLLM/Mooncake 完成 abort final-save 和 aggregate barrier；dynamic controller 不直接调用 Mooncake API，也不接收逐请求 hit/miss 结果。

consumer connector 自行通过 lookup hit/miss 选择 GET 或完整 recompute。sleep 路径把已有 `_reset_caches(reset_connector=False)` 接入公共 `sleep()`，清理 local KV 而保留外部 store。首版不实现 capability negotiation、`policy_version` 参数或 `before_device_sleep()`；后者只作为未来 FlexKV 扩展注释保留。

对于 SGLang，只复用 `abort_replicas(checkpoint_kv=True)` 的 barrier 语义，由 SGLang engine adapter 在 HiCache 内实现 checkpoint。Mooncake 能自然映射到 HiCache L3；FlexKV 当前仍需额外 SGLang storage 集成。

## 12. 待讨论问题

已确认：

- 外部 KV 达到可 lookup 状态且 connector 不再访问 source GPU 后，aggregate barrier 返回；retry 不等待 hybrid sleep 完成。
- 不建立新的 VERL lifecycle abstraction，只扩展现有 `abort_replicas(checkpoint_kv=True)`。
- VERL 不接收逐请求 `reusable/fallback`；consumer connector 通过 hit/miss 自动决定 GET 或 recompute。
- 首版固定同构 Mooncake 配置，不实现通用 capability negotiation。
- 首版采用单 policy invariant，不传 `policy_version`；多 policy overlap 时再增加 namespace。
- Mooncake 首版不实现 `before_device_sleep()`，只保留未来 FlexKV 注释。
- 公共 sleep 路径使用已有 `_reset_caches(reset_connector=False)`，这是接线改动，不是新增内部 reset 接口。

其余待讨论问题：

1. 计划使用的 FlexKV 版本是否提供区别于 transfer completion 的全局可见性确认？
2. SGLang HiCache 能否显式 flush/pin 指定 aborted request 的 radix node，还是必须增加 scheduler API？
3. 第一版 SGLang + FlexKV 是否允许 CPU staging？
