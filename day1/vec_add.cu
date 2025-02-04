#include <cstdint>
#include <cstdlib>
#include <stdio.h>
#include <time.h>

double diff;
struct timespec start_time, end_time;

double time_diff(struct timespec end_time, struct timespec start_time) {
  double diff = (end_time.tv_sec - start_time.tv_sec) +
                (end_time.tv_nsec - start_time.tv_nsec) / 1000000000.0;
  return diff;
}

void vec_add_cpu(int32_t *x, int32_t *y, int32_t *z, int32_t N) {
  for (uint32_t i = 0; i < N; ++i) {
    z[i] = x[i] + y[i];
  }
}

__global__ void vecadd_kernel(int32_t *x, int32_t *y, int32_t *z, int32_t N) {
  uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= N)
    return;
  z[i] = x[i] + y[i];
}

void vec_add_gpu(int32_t *x, int32_t *y, int32_t *z, int32_t N) {
  uint32_t numBlocks, numThreads;
  int32_t *x_d, *y_d, *z_d;

  // allocate memory
  clock_gettime(CLOCK_MONOTONIC, &start_time);
  cudaMalloc(&x_d, N * sizeof(int32_t));
  cudaMalloc(&y_d, N * sizeof(int32_t));
  cudaMalloc(&z_d, N * sizeof(int32_t));

  // copy vector to GPU
  cudaMemcpy(x_d, x, N, cudaMemcpyHostToDevice);
  clock_gettime(CLOCK_MONOTONIC, &end_time);
  diff = time_diff(end_time, start_time);
  printf("CPU to GPU copy time: %f\n", diff);

  // perform addition
  numThreads = 512;
  numBlocks = (N + numThreads - 1) / numThreads;
  vecadd_kernel<<<numBlocks, numThreads>>>(x_d, y_d, z_d, N);

  // copy result to CPU
  cudaMemcpy(z, z_d, N, cudaMemcpyDeviceToHost);

  // free memory
  cudaFree(x_d);
  cudaFree(y_d);
  cudaFree(z_d);
}

int32_t main() {
  cudaDeviceSynchronize();

  int32_t N = 100000000;

  // read data
  int32_t *x = (int32_t *)malloc(N * sizeof(int32_t));
  int32_t *y = (int32_t *)malloc(N * sizeof(int32_t));
  int32_t *z = (int32_t *)malloc(N * sizeof(int32_t));
  for (uint32_t i = 0; i < N; ++i) {
    x[i] = rand();
    y[i] = rand();
  }

  // cpu
  clock_gettime(CLOCK_MONOTONIC, &start_time);
  vec_add_cpu(x, y, z, N);
  clock_gettime(CLOCK_MONOTONIC, &end_time);
  diff = time_diff(end_time, start_time);
  printf("CPU time: %f\n", diff);

  // gpu
  clock_gettime(CLOCK_MONOTONIC, &start_time);
  vec_add_gpu(x, y, z, N);
  cudaDeviceSynchronize();
  clock_gettime(CLOCK_MONOTONIC, &end_time);
  diff = time_diff(end_time, start_time);
  printf("GPU time: %f\n", diff);

  free(x);
  free(y);
  free(z);
}
