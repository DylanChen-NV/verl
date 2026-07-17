#!/usr/bin/env bash
set -euo pipefail

runtime_dir=${1:-/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/05_claude_ws/dynamic_mooncake_abort_dev/runtime}
cuda_root=${CUDA_HOME:-/usr/local/cuda}

gcc -shared -fPIC -O2 -Wall -Wextra \
  -I"${cuda_root}/include" \
  "${runtime_dir}/cuda_memcpy_device_guard.c" \
  -L"${cuda_root}/lib64" -Wl,-rpath,"${cuda_root}/lib64" \
  -lcudart -ldl -pthread \
  -o "${runtime_dir}/libcuda_memcpy_device_guard.so"

gcc -O2 -Wall -Wextra \
  -I"${cuda_root}/include" \
  "${runtime_dir}/test_cuda_memcpy_device_guard.c" \
  -L"${cuda_root}/lib64" -Wl,-rpath,"${cuda_root}/lib64" \
  -lcudart -pthread \
  -o "${runtime_dir}/test_cuda_memcpy_device_guard"

echo "without_preload"
"${runtime_dir}/test_cuda_memcpy_device_guard" || true
echo "with_preload"
LD_PRELOAD="${runtime_dir}/libcuda_memcpy_device_guard.so" \
  "${runtime_dir}/test_cuda_memcpy_device_guard"
