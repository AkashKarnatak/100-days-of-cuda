#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

struct timer {
  struct timespec start_time, end_time;
};

void start_timer(struct timer *t) {
  clock_gettime(CLOCK_MONOTONIC, &t->start_time);
}

void stop_timer(struct timer *t) {
  clock_gettime(CLOCK_MONOTONIC, &t->end_time);
}

double time_diff(struct timer *t) {
  double diff = (t->end_time.tv_sec - t->start_time.tv_sec) +
                (t->end_time.tv_nsec - t->start_time.tv_nsec) / 1000000000.0;
  return diff;
}

struct timer t;

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
  start_timer(&t);
  cudaMalloc(&x_d, N * sizeof(int32_t));
  cudaMalloc(&y_d, N * sizeof(int32_t));
  cudaMalloc(&z_d, N * sizeof(int32_t));

  // copy vector to GPU
  cudaMemcpy(x_d, x, N, cudaMemcpyHostToDevice);
  stop_timer(&t);
  printf("CPU to GPU copy time: %f\n", time_diff(&t));

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
  start_timer(&t);
  vec_add_cpu(x, y, z, N);
  stop_timer(&t);
  printf("CPU time: %f\n", time_diff(&t));

  // gpu
  start_timer(&t);
  vec_add_gpu(x, y, z, N);
  cudaDeviceSynchronize();
  stop_timer(&t);
  printf("GPU time: %f\n", time_diff(&t));

  free(x);
  free(y);
  free(z);
}
