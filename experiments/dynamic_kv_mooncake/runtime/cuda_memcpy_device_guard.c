#define _GNU_SOURCE

#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <pthread.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

typedef cudaError_t (*cuda_memcpy_fn)(void *, const void *, size_t,
                                      enum cudaMemcpyKind);
typedef cudaError_t (*cuda_memcpy_async_fn)(void *, const void *, size_t,
                                            enum cudaMemcpyKind,
                                            cudaStream_t);

static pthread_once_t resolve_once = PTHREAD_ONCE_INIT;
static cuda_memcpy_fn real_cuda_memcpy;
static cuda_memcpy_async_fn real_cuda_memcpy_async;

static void resolve_symbols(void) {
  real_cuda_memcpy = (cuda_memcpy_fn)dlsym(RTLD_NEXT, "cudaMemcpy");
  real_cuda_memcpy_async =
      (cuda_memcpy_async_fn)dlsym(RTLD_NEXT, "cudaMemcpyAsync");
}

static int pointer_device(const void *ptr) {
  struct cudaPointerAttributes attrs;
  cudaError_t status = cudaPointerGetAttributes(&attrs, ptr);
  if (status != cudaSuccess) {
    (void)cudaGetLastError();
    return -1;
  }
#if CUDART_VERSION >= 10000
  return attrs.type == cudaMemoryTypeDevice ? attrs.device : -1;
#else
  return attrs.memoryType == cudaMemoryTypeDevice ? attrs.device : -1;
#endif
}

static int transfer_device(void *dst, const void *src,
                           enum cudaMemcpyKind kind) {
  if (kind == cudaMemcpyDeviceToHost) {
    return pointer_device(src);
  }
  if (kind == cudaMemcpyHostToDevice) {
    return pointer_device(dst);
  }
  int device = pointer_device(src);
  return device >= 0 ? device : pointer_device(dst);
}

static int switch_device_for_transfer(void *dst, const void *src,
                                      enum cudaMemcpyKind kind,
                                      int *previous_device) {
  if (cudaGetDevice(previous_device) != cudaSuccess) {
    (void)cudaGetLastError();
    *previous_device = -1;
  }
  int device = transfer_device(dst, src, kind);
  if (device < 0 || device == *previous_device) {
    return 0;
  }
  if (cudaSetDevice(device) != cudaSuccess) {
    (void)cudaGetLastError();
    return 0;
  }
  return 1;
}

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
                       enum cudaMemcpyKind kind) {
  pthread_once(&resolve_once, resolve_symbols);
  if (real_cuda_memcpy == NULL) {
    return cudaErrorSharedObjectSymbolNotFound;
  }
  int previous_device = -1;
  int switched =
      switch_device_for_transfer(dst, src, kind, &previous_device);
  cudaError_t status = real_cuda_memcpy(dst, src, count, kind);
  if (switched && previous_device >= 0) {
    (void)cudaSetDevice(previous_device);
  }
  return status;
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
                            enum cudaMemcpyKind kind, cudaStream_t stream) {
  pthread_once(&resolve_once, resolve_symbols);
  if (real_cuda_memcpy_async == NULL) {
    return cudaErrorSharedObjectSymbolNotFound;
  }
  int previous_device = -1;
  int switched =
      switch_device_for_transfer(dst, src, kind, &previous_device);
  cudaError_t status = real_cuda_memcpy_async(dst, src, count, kind, stream);
  if (switched && previous_device >= 0) {
    (void)cudaSetDevice(previous_device);
  }
  return status;
}
