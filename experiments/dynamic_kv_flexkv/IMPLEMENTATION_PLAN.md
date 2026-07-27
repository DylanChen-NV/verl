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

## Acceptance

