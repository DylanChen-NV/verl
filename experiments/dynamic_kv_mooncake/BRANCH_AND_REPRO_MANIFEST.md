# Dynamic KV Mooncake Branch And Reproduction Manifest

## Branch topology

- `dynamic-kv-backend-dev`: backend-neutral dynamic abort, visibility barrier,
  retry gate, and connector-preserving sleep lifecycle. Baseline commit:
  `c47e490`.
- `dynamic-kv-mooncake`: Mooncake connector validation, vLLM patch, runners,
  analyzers, tests, and the accepted A/B/C evidence. Use this branch to
  reproduce the completed Mooncake work.
- `dynamic-kv-flexkv`: starts from `dynamic-kv-backend-dev`. FlexKV work must
  not inherit the old conflicting experimental patch.

The earlier `mooncake-abort-reuse` branch is retained as historical pre-final
state and is not the reproduction entry point.

## Fixed runtime

- DFW, 2 nodes x 8 H100
- image `verlai/verl:vllm023.dev1`
- Qwen3-30B-A3B BF16
- trainer Megatron TP2/PP2/EP4
- rollout vLLM TP2, sleep enabled
- DAPO-Math-17K, response length 8192
- `rollout.total_rollout_steps=80`
- `concurrent_samples_per_replica=32`
- `rollout.max_num_seqs=32`

## Entry points

- A: `scripts/run_persistent_moon_a_formal_r80.sh`
- B: `scripts/run_persistent_moon_b_formal_r80.sh`
- C: `scripts/run_persistent_moon_c_formal_r80_r2.sh`
- postprocess: `scripts/postprocess_mooncake_formal_run.sh`
- strict proof: `scripts/analyze_mooncake_strict_reuse.py`
- target comparison: `scripts/compare_mooncake_target_cohorts.py`

## Patches

- `patches/verl_kv_backend_validation.patch`
  SHA256 `4019c9bebd8302b394c636d7eee1db6622175a93b367a6dbf8edd39fc65d39f8`
- `patches/vllm_mooncake_abort_reuse.patch`
  SHA256 `6b032614e0d97cfe9a6b48e2d26bef0d5d22eb4a158217b4bc856b8ede040583`

## Accepted result

The formal C run proved 26/26 strict cross-node retries, with 99.895% mean KV
hit ratio and about 89.5% mean queue-excluded recompute reduction. The current
