#include <math.h>
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

uint32_t cdiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

__global__ void gelu_kernel(float *A, float *B, uint32_t N) {
  uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= N)
    return;
  B[i] = 0.5f * A[i] * (1.0f + erff(A[i] / sqrtf(2.0f)));
}

void gelu_gpu(float *A, float *B, uint32_t N) {
  float *A_d, *B_d;

  // allocate memory on GPU
  cudaMalloc(&A_d, N * sizeof(float));
  cudaMalloc(&B_d, N * sizeof(float));

  // copy data from CPU to GPU
  cudaMemcpy(A_d, A, N * sizeof(float), cudaMemcpyHostToDevice);

  // perform computation
  int32_t numThreads = 1024;
  int32_t numBlocks = cdiv(N, numThreads);
  start_timer(&t);
  gelu_kernel<<<numBlocks, numThreads>>>(A_d, B_d, N);
  cudaDeviceSynchronize();
  stop_timer(&t);
  printf("GPU time: %f\n", time_diff(&t));

  // copy data back to CPU
  cudaMemcpy(B, B_d, N * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(A_d);
  cudaFree(B_d);
}

void gelu_cpu(float *A, float *B, uint32_t N) {
  start_timer(&t);
  for (uint32_t i = 0; i < N; ++i) {
    B[i] = 0.5f * A[i] * (1.0f + erff(A[i] / sqrtf(2.0f)));
  }
  stop_timer(&t);
  printf("CPU time: %f\n", time_diff(&t));
}

bool all_close(float *A, float *B, uint32_t N) {
  for (uint32_t i = 0; i < N; ++i) {
    if (abs(A[i] - B[i]) >= 1e-5)
      return false;
  }
  return true;
}

void print_arr(float *A, uint32_t N) {
  printf("[\n");
  for (uint32_t i = 0; i < N; ++i) {
    printf(" %f", A[i]);
  }
  printf("\n]\n");
}

int main() {
  cudaDeviceSynchronize();

  uint32_t N = 1'00'000'000;
  float *A, *B, *C;

  A = (float *)malloc(N * sizeof(float));
  B = (float *)malloc(N * sizeof(float));
  C = (float *)malloc(N * sizeof(float));

  for (uint32_t i = 0; i < N; ++i) {
    A[i] = (float)rand() / RAND_MAX;
  }

  gelu_cpu(A, B, N);

  gelu_gpu(A, C, N);

  // print_arr(B, N);
  // print_arr(C, N);

  printf("GPU impl match: %s\n", all_close(B, C, N) ? "true" : "false");

  free(A);
  free(B);
}
