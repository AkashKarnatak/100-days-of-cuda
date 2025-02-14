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

__global__ void conv1D_kernel(float *A, float *B, float *K, size_t N,
                              size_t S) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= N)
    return;

  float sum = 0.0f;
  for (size_t j = 0; j < S; ++j) {
    sum += K[j] * A[i + j];
  }
  B[i] = sum;
}

void conv1D_gpu(float *A, float *B, float *K, size_t N, size_t N_out,
                size_t S) {
  float *A_d, *B_d, *K_d;

  // allocate memory on GPU
  cudaMalloc(&A_d, N * sizeof(float));
  cudaMalloc(&B_d, N_out * sizeof(float));
  cudaMalloc(&K_d, S * sizeof(float));

  // copy data from CPU to GPU
  cudaMemcpy(A_d, A, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(K_d, K, S * sizeof(float), cudaMemcpyHostToDevice);

  // perform computation
  int32_t numThreads = 1024;
  int32_t numBlocks = cdiv(N_out, numThreads);
  start_timer(&t);
  conv1D_kernel<<<numBlocks, numThreads>>>(A_d, B_d, K_d, N, S);
  cudaDeviceSynchronize();
  stop_timer(&t);
  printf("GPU time: %f\n", time_diff(&t));

  // copy data back to CPU
  cudaMemcpy(B, B_d, N_out * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(A_d);
  cudaFree(B_d);
}

void conv1D_cpu(float *A, float *B, float *K, size_t N, size_t N_out,
                size_t S) {
  start_timer(&t);
  for (size_t i = 0; i < N_out; ++i) {
    float sum = 0.0f;
    for (size_t j = 0; j < S; ++j) {
      sum += A[i + j] * K[j];
    }
    B[i] = sum;
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

  size_t N = 1'00'000'000;
  size_t S = 100;
  size_t N_out = N - S + 1;

  float *A, *B, *C, *K;

  A = (float *)malloc(N * sizeof(float));
  B = (float *)malloc(N_out * sizeof(float));
  C = (float *)malloc(N_out * sizeof(float));
  K = (float *)malloc(S * sizeof(float));

  for (size_t i = 0; i < N; ++i) {
    A[i] = (float)rand() / RAND_MAX;
  }
  for (size_t i = 0; i < S; ++i) {
    K[i] = (float)rand() / RAND_MAX;
  }

  conv1D_cpu(A, B, K, N, N_out, S);

  conv1D_gpu(A, C, K, N, N_out, S);

  // print_arr(B, N);
  // print_arr(C, N);

  printf("GPU impl match: %s\n", all_close(B, C, N_out) ? "true" : "false");

  free(A);
  free(B);
}
