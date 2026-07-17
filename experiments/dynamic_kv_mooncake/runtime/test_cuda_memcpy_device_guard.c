#include <cuda_runtime_api.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

struct copy_args {
  const int *device_ptr;
  int value;
  cudaError_t status;
};

static void *copy_from_wrong_device(void *opaque) {
  struct copy_args *args = (struct copy_args *)opaque;
  if (cudaSetDevice(0) != cudaSuccess) {
    args->status = cudaGetLastError();
    return NULL;
  }
  args->status = cudaMemcpy(&args->value, args->device_ptr, sizeof(int),
                            cudaMemcpyDeviceToHost);
  return NULL;
}

int main(void) {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count < 2) {
    fprintf(stderr, "requires at least two CUDA devices\n");
    return 2;
  }
  if (cudaSetDevice(1) != cudaSuccess) {
    return 3;
  }
  int *device_ptr = NULL;
  int expected = 0x12345678;
  if (cudaMalloc((void **)&device_ptr, sizeof(int)) != cudaSuccess ||
      cudaMemcpy(device_ptr, &expected, sizeof(int), cudaMemcpyHostToDevice) !=
          cudaSuccess) {
    return 4;
  }

  struct copy_args args = {.device_ptr = device_ptr,
                           .value = 0,
                           .status = cudaSuccess};
  pthread_t thread;
  if (pthread_create(&thread, NULL, copy_from_wrong_device, &args) != 0) {
    return 5;
  }
  pthread_join(thread, NULL);
  cudaSetDevice(1);
  cudaFree(device_ptr);
  printf("status=%d value=%#x expected=%#x\n", (int)args.status, args.value,
         expected);
  return args.status == cudaSuccess && args.value == expected ? 0 : 1;
}
