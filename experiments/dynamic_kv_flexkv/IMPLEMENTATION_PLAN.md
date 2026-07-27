# Dynamic Resource FlexKV Abort-Reuse Implementation Plan

Date: 2026-07-17

## Objective

In the sleep-enabled VERL dynamic-resource flow, offload partial KV from
aborted hybrid rollout requests and let their retries reuse it from standalone
rollout replicas, including strict cross-node reuse.

## Fixed branch baselines

- VERL: `dynamic-kv-flexkv`, initially identical to the accepted Mooncake
  commit `d177865`.
- vLLM: `dynamic-kv-flexkv`, based on `dc9f845`, with connector-preserving
  sleep `8fe5ea9` and the original-author FlexKV reset commits cherry-picked as
  `b553daf` and `befc1aa`.
- FlexKV: create `dynamic-kv-flexkv` from
  `jianyingzhu/FlexKV feat/rl_kv_clear@dbd912e` without copying or squashing
  upstream author history.

Mooncake branches remain unchanged. No SSD work is in scope.

## Ownership

- VERL orchestrates dynamic deactivate, retry gating, sleep, wake, parameter
  synchronization, and policy-boundary reset.
- vLLM owns the policy decision whether an aborted request must be offloaded.
- Mooncake and FlexKV declare support and execute the vLLM directive; they do
  not read policy environment variables or decide whether offload is useful.

## vLLM changes

1. Add `offload_aborted_kv: bool | None = None` to
   `AsyncLLM.pause_generation()` and propagate it through EngineCore client,
   EngineCore, and Scheduler abort handling.
2. Resolve the value in vLLM. An explicit argument wins; the temporary test
   override `VLLM_ABORT_KV_OFFLOAD=0|1` is read only in vLLM; `None` defaults
   to enabled when the configured connector declares abort-offload support.
3. Store the resolved command on each request as a finish directive. Keep the
   existing connector `request_finished()` signature to minimize churn.
4. Add a connector capability for abort KV offload. Mooncake Store and FlexKV
   declare support.
5. When offload is requested, do not complete pause until connector PUT and
   delayed-free cleanup complete. First validate the existing
   `has_finished_requests()` path; add `has_pending_push_work()` only if needed.
6. Replace Mooncake's backend-local environment-variable policy with the vLLM
   finish directive after compatibility tests pass.

## FlexKV changes

1. In `FlexKVSchedulerConnector.request_finished()`, allow
   `FinishReason.ABORT` to enter the existing PUT path only when the vLLM
   request finish directive asks for offload.
2. Reuse `_put_match()`, asynchronous PUT, `finished_sending`, and delayed-free
   behavior. Do not add a FlexKV policy environment variable.
3. Add focused tests for partial generation -> abort -> PUT -> retry external
   hit, plus offload-disabled behavior.

Do not modify reset RPC, KVManager reset, initial VMM registration,
TransferManager registration, SSD support, or metrics in the first version.

## VERL changes

1. Generalize backend validation to accept both
   `MooncakeStoreConnector` and `FlexKVConnectorV1` with role `kv_both`.
2. Add the FlexKV engine/runtime configuration and test-only
   `VLLM_ABORT_KV_OFFLOAD` override.
3. Reuse the accepted sequence: close retry gate -> remove hybrid replicas ->
   abort/offload -> wait for visibility -> release retry gate ->
   `sleep(reset_connector=False)` -> retry.
4. Keep `reset_connector=True` at the policy/parameter-sync boundary so stale
   policy KV cannot be reused.

## Validation order

1. Static checks and focused unit tests for the new vLLM directive and FlexKV
   aborted PUT decision.
2. Single-GPU vLLM + FlexKV E2E: offload disabled/enabled, delayed-free barrier,
   reset miss, and connector-preserving sleep.
3. Validate VMM handle correctness across sleep/wake. If it fails, record the
   failure and a re-registration TODO; do not implement re-registration in
   this iteration.
4. VERL single-node dynamic smoke test.
5. Two-node strict proof: hybrid producer request/key/IP -> standalone consumer
   receive/key/IP -> successful first output, with forced-miss control.
6. Reuse the Mooncake A/B/C analysis to report hit ratio, queue-excluded
   recompute reduction, PUT-visible barrier, GET overhead, and net cycle gain.

## Progress (2026-07-17)

### Isolated commits

- vLLM `dynamic-kv-flexkv`: `9c2d6dc`; upstream FlexKV commits remain
  separate with their original authors.
- FlexKV `dynamic-kv-flexkv`: directive commit `a687c9e`, packed-layout and E2E commit `2309434`.
- VERL `dynamic-kv-flexkv`: backend validation and this record are committed
  together after verification.

### Fixed DFW debug environment

- Session `dyn-flexkv-abort-dev-20260717`, job `14105065`, 2 nodes x 8 H100,
  `pool0-[01698,01701]`; do not release without explicit approval.
- Image `verlai/verl:vllm023.dev1`.
- Workdir:
  `/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/05_claude_ws/dynamic_flexkv_abort_dev`.
- Clones: vLLM branch `feat/rl_flexkv`, commit `745d3f3`; FlexKV
  branch `feat/rl_kv_clear`, commit `dbd912e`.
  Apply connector-preserving sleep with `git am`,
  then apply the isolated development diffs.

### Environment preparation

1. Install `transformers==5.13.1`, `tblib`, build helpers, and
   `liburing-dev`.
2. Install the branch in editable mode with
   `VLLM_USE_PRECOMPILED=1 python3 -m pip install --no-build-isolation -e .`.
3. Align FlashInfer with this vLLM revision:
   `python3 -m pip install --extra-index-url https://flashinfer.ai/whl/ flashinfer-cubin==0.6.14`.
4. Build FlexKV with
   `FLEXKV_ENABLE_METRICS=0 MAX_JOBS=8 ./build.sh`; the resulting
   `flexkv/c_ext.so` imports successfully. SSD, nvcomp, metrics, P2P, and
   Redis remain disabled.

Use a short, pre-created `VLLM_RPC_BASE_PATH` under `/tmp`; a long IPC path
or a missing directory prevents EngineCore startup.

### Additional compatibility work

The supplied FlexKV adapter expected the old five-dimensional split K/V
tensor. vLLM commit `51878e5` now supplies packed four-dimensional tensors
such as `[blocks, heads, block_size, 2 * head_size]`. FlexKV now:

- detects the backend shape from vLLM config;
- records `packed_kv=True` without treating the model as MLA;
- allocates CPU mirrors with one packed K/V region and unchanged byte size;
- registers both current 4D packed and legacy 5D split layouts;
- sends packed blocks through the existing transfer kernel as one contiguous
  region. No new CUDA kernel was required.

### Completed validation

- vLLM focused control tests: `9 passed`; scheduler integration test with the
  local Qwen3-8B config: `1 passed`.
- FlexKV finish-decision tests: `3 passed`.
- Python 3.12 compile and `git diff --check` pass for the packed-layout
  changes.
- Non-sleep external-cache gate:
  `FLEXKV_TEST_ENABLE_SLEEP=0 FLEXKV_CPU_CACHE_GB=8 pytest -q -s tests/test_reset_cache_vllm_e2e.py::test_flexkv_populates_before_asserting_reset`
  passed. Eight long prompts produced about 3.5 GiB of D2H traffic; after only
  local prefix cache was reset, retry produced seven H2D transfers and an
  external hit.
- Abort partial-trajectory gate:
  `FLEXKV_CPU_CACHE_GB=2 pytest -q -s tests/test_abort_reuse_vllm_e2e.py::test_aborted_partial_trajectory_reuses_flexkv`
  passed. The request was aborted after 33 generated tokens; pause returned
  only after a 0.279 GiB D2H save; retry used original prompt tokens plus the
  33 partial tokens, forced local-cache cold, performed a 0.279 GiB H2D load,
  and reported `2032/2039` external hits (99.66%).

These results prove the vLLM directive, delayed-free barrier, FlexKV offload,
and partial trajectory reuse in non-sleep mode.

### Sleep/VMM validation result

Sleep-enabled startup reaches packed-layout detection but fails while
registering the CuMemAllocator-backed tensor:

`RuntimeError: VMM allocation has no shareable handle type set (requestedHandleTypes=0x0)`.

On the DFW H100, vLLM's `csrc/cumem_allocator.cpp` requests a Fabric handle
only when the device advertises Fabric support. When it does not, the initial
allocation requests no shareable handle; POSIX FD is selected only as a
fallback from a failed Fabric allocation. FlexKV therefore cannot export the
allocation. This was the original VMM validation finding; the temporary opt-in allocator
change and its validation are recorded below.

One-line summary: in vLLM sleep mode the allocation is currently created as
not shareable across processes; Mooncake does not need sharing because it uses
the tensor in the vLLM worker process, while FlexKV accesses it from another
process and therefore fails.

Temporary bring-up decision: add an opt-in environment variable in vLLM to
request a shareable VMM handle for sleep-mode allocations, so the FlexKV path
can be validated without changing the default allocator behavior. The temporary variable is
`VLLM_CUMEM_ENABLE_SHAREABLE_HANDLE=1`. TODO: replace this test
switch with a normal vLLM parameter/capability that is enabled automatically
when `FlexKV + sleep mode` requires cross-process tensor sharing.

### Validation update (2026-07-17)

The temporary vLLM switch is `VLLM_CUMEM_ENABLE_SHAREABLE_HANDLE=1`. With
this switch on a non-Fabric DFW H100:

- `min-vllm-flexkv-export-r1` passed: a vLLM CuMemAllocator tensor exported as
  `vmm_posix_fd`;
- `flexkv-compact-sleep-cycle-r3` passed with `rc=0`: all 36 Qwen3-8B KV
  tensors exported, the FlexKV process imported all POSIX FD handles, initial
  generation passed, `sleep(level=1)` passed, `wake_up()` passed, and
  generation after wake passed.

This clears the original initialization blocker and validates the basic
sleep/wake lifecycle. The environment variable is intentionally temporary and
currently produces an unknown-vLLM-environment-variable warning because it is
read directly by the C++ allocator. Registering or replacing it with a formal
vLLM parameter remains part of the refinement TODO.

### Remaining validation

1. Run the VERL dynamic single-node smoke: deactivate hybrid -> abort/offload
   barrier -> sleep -> wake/retry.
2. Run the strict two-node producer/consumer proof with request/key/node
   correlation and a forced-miss control.
3. Run A/B/C trajectory analysis and report hit ratio, queue-excluded
   recompute reduction, PUT barrier, GET overhead, and net cycle gain.

## Acceptance

The feature is complete only when the sleep-enabled dynamic flow and strict
cross-node proof pass. The current state has passed non-sleep abort trajectory
reuse plus the vLLM/FlexKV initialization and basic sleep/wake gate. VERL
dynamic integration, strict cross-node reuse, and A/B/C remain pending.


### VERL multi-instance bring-up update (2026-07-17)

Fixed DFW session `dyn-flexkv-verl-dev-20260717`, job `14110162`, nodes
`pool0-[00866,00885]` remains allocated and must not be released without
explicit approval.

- `r5` proved that four TP2 rollout engines cannot use the default FlexKV
  single-instance mode: every engine used the same IPC endpoint and waited for
  an independent 2-GPU registration.
- VERL now maps its existing `replica_rank` to
  `FLEXKV_INSTANCE_ID = replica_rank % FLEXKV_INSTANCE_NUM` only when
  `FLEXKV_INSTANCE_NUM > 1`. The isolated runner sets
  `FLEXKV_INSTANCE_NUM=4` and a run-specific `FLEXKV_SERVER_RECV_PORT`.
- `r8` proved that this mapping reaches the intended FlexKV topology:
  scheduler/worker logs show `instance_id=0,1,2,3`,
  `dp_client_id=0,1,2,3`, one shared KVServer, and `expected_gpus=8`.
- `r8` then stopped at `2/8` GPU registrations. Every vLLM instance still
  registered TP-local `device_id=0,1`; the server rejected later requests as
  duplicates. This is not a VERL lifecycle failure.

The agreed fix remains FlexKV-only, but must not overload one `device_id` as
both a logical registration identity and a CUDA device locator. Reuse the
existing FlexKV rank model and define:

```text
intra_client_id = pp_rank * effective_tp_size + effective_tp_rank
registration_key = (dp_client_id, intra_client_id)
```

`dp_client_id` uniquely identifies the replica/client. `effective_tp_rank`
already folds the TP/attention-CP data-plane rank, so PP/TP/CP do not need to
be repeated as separate fields in the registration key. Do not flatten
`dp_client_id` and `intra_client_id` into one global integer: FlexKV still
needs `dp_client_id` directly for replica-level routing and cleanup.

Keep the node-local CUDA device locator as separate registration metadata:

```text
registration_key -> logical GPU worker / registered KV handles
local_device_id   -> torch.cuda.set_device(), VMM import, local GPU access
```

Obtain the local/physical mapping through vLLM's platform API rather than
parsing `CUDA_VISIBLE_DEVICES`. `TransferManager` should index individual GPU
registrations by `registration_key`, while retaining its existing
`WorkerKey(dp_client_id, pp_rank)` grouping for transfer-engine operations.
The local device ID must not be treated as globally unique across nodes.

Implementation and validation order:

1. Extend the FlexKV registration request and maps to carry
   `registration_key` separately from `local_device_id`.
2. Update the FlexKV vLLM adapter to derive `intra_client_id` from `RankInfo`
   and obtain the CUDA locator through the vLLM platform API.
3. Add a focused 4-instance x TP2 registration test, then rerun the same VERL
   dynamic smoke.
4. Follow with TP16 and multi-node multi-replica coverage; FlexKV's documented
   multi-instance support is currently single-node, so multi-node support is
   not considered proven by the first smoke.

Forcibly interrupted vLLM/FlexKV runs can leave EngineCore, GPU worker, and
MPS processes outside Ray; clean those processes and verify
`torch.cuda.device_count()==8` on both persistent ranks before rerunning.

### Node-level shared-server ownership update (2026-07-18)

The registration-key change passed the VERL 4-instance x TP2 registration
barrier: all eight GPU workers registered as `(dp_client_id,
intra_client_id)`. The next failure was in the data plane:

```text
device=2, num_gpus=2
```

The original multi-instance mode starts the shared `KVServer` from
`dp_client_id=0`. Under VERL/Ray that client belongs to one TP2 vLLM actor,
so the child server inherits a process scope that can access only those two
GPUs. Importing VMM handles from the other three replicas therefore fails even
though their logical registrations are correct.

Agreed topology and ownership:

1. Run one FlexKV server per physical rollout node; the server must see all
   GPUs on that node.
2. Every vLLM instance on the node, including `dp_client_id=0`, is a client of
   that server.
3. Dynamic activate/deactivate and vLLM sleep/wake do not stop the node-level
   server. The node owner stops it only when the VERL job exits.
4. Preserve FlexKV's default embedded mode for ordinary vLLM serving.
5. Use `FLEXKV_SERVER_LAUNCH_MODE=external` as the temporary bring-up switch.
   In external mode `KVManager` neither creates nor shuts down the shared
   server; client shutdown only unregisters that client.

TODO(parameterization): replace `FLEXKV_SERVER_LAUNCH_MODE` environment
selection with an explicit vLLM/VERL configuration parameter. VERL should pass
the resolved ownership mode through the connector configuration; FlexKV should
execute that instruction rather than independently infer deployment policy.
Keep the environment variable only as a temporary experiment override until
that parameter path is implemented.

Validation order:

1. FlexKV unit test: embedded behavior unchanged; external client 0 never
   creates/stops the server; unregister leaves the server running.
2. FlexKV-only node test: one externally owned server plus four TP2 clients,
   with `torch.cuda.device_count()==8` in the server process.
3. VERL dynamic smoke: node server starts before vLLM clients, all 8/8 workers
   register, transfer workers initialize, and sleep/wake completes.
4. Distributed-index/transfer setup and strict cross-node reuse proof.


## Two-node external-server validation (2026-07-18)

- A stale FlexKV TransferManager multiprocessing tree from an earlier run held about 27.3 GiB per GPU. Always audit both nodes with `nvidia-smi --query-compute-apps` before a rerun.
- Clean baseline: 0 MiB per GPU. A waiting external FlexKV server plus node-level MPS uses about 78 MiB per GPU; it does not allocate or import vLLM KV cache before registration.
- Dynamic hybrid requires one external FlexKV server on every rollout-capable physical node, including trainer nodes that may later activate hybrid replicas. Node-local IPC paths may use the same name on different hosts.
- `flexexternal5` proved independent registration on both nodes: each server received four DP clients and registration keys `(0,0)` through `(3,1)`, mapped to physical devices 0 through 7, then reported `All 8 GPUs registered successfully`.
- Launch each server in its own process group (`setsid`) and terminate the process group during cleanup. The post-run audit was 0 MiB on all 16 GPUs with no compute processes.
- The next blocker is capacity during the initial parameter sync, not server ownership: with `gpu_memory_utilization=0.70`, the 2 GiB bucketed weight-transfer buffer saw only 1.86 GiB free. Next test: make the existing hard-coded utilization overrideable and run at 0.65 before considering code changes.
- TODO(parameterization): replace `FLEXKV_SERVER_LAUNCH_MODE` and other temporary environment switches with explicit vLLM/VERL connector configuration after bring-up.
