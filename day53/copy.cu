#include <assert.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <time.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

__device__ __host__ inline size_t cdiv(size_t a, size_t b) {
  return (a + b - 1) / b;
}

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

__global__ void copy_kernel(float *in, float *out, size_t N, size_t M) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (4 * idx >= N * M)
    return;

  float4 in4 = ((float4 *)in)[idx];
  ((float4 *)out)[idx] = in4;
}

bool allclose(float *A, float *B, size_t N, size_t M) {
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < M; ++j) {
      if (abs(A[i * M + j] - B[i * M + j]) > 1e-4) {
        printf("Mismatch at (%lu, %lu), A = %f and B = %f\n", i, j,
               A[i * M + j], B[i * M + j]);
        return false;
      }
    }
  }
  return true;
}

void print_mat(float *A, size_t N, size_t M) {
  printf("[\n");
  for (size_t row = 0; row < N; ++row) {
    for (size_t col = 0; col < M; ++col) {
      printf("  %f ", A[row * M + col]);
    }
    printf("\n");
  }
  printf("]\n");
}

void copy_cpu(float *in, float *out, size_t N, size_t M) {
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < M; ++j) {
      out[i * M + j] = in[i * M + j];
    }
  }
}

int main() {
  size_t N, M;
  float *in, *out, *out_base;
  float *in_d, *out_d;
  dim3 numThreads, numBlocks;

  N = M = 16384;

  in = (float *)malloc(N * M * sizeof(float));
  out_base = (float *)malloc(N * M * sizeof(float));
  out = (float *)malloc(N * M * sizeof(float));

  cudaMalloc(&in_d, N * M * sizeof(float));
  cudaMalloc(&out_d, N * M * sizeof(float));

  for (size_t i = 0; i < N * M; ++i) {
    in[i] = (float)rand() / RAND_MAX;
  }

  copy_cpu(in, out_base, N, M);

  CUDA_CHECK(
      cudaMemcpy(in_d, in, N * M * sizeof(float), cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaDeviceSynchronize());
  cudaMemset(out_d, 0, N * M * sizeof(float));
  start_timer(&t);
  const size_t BLOCK_SIZE = 16;
  numThreads = dim3(BLOCK_SIZE * BLOCK_SIZE);
  numBlocks = dim3(cdiv(N * M, numThreads.x * 4));
  copy_kernel<<<numBlocks, numThreads>>>(in_d, out_d, N, M);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  stop_timer(&t);
  CUDA_CHECK(
      cudaMemcpy(out, out_d, N * M * sizeof(float), cudaMemcpyDeviceToHost));
  printf("Naive copy kernel time: %f\n", time_diff(&t));
  printf("Match impl: %s\n\n",
         allclose(out_base, out, M, N) ? "true" : "false");

  cudaFree(in_d);
  cudaFree(out_d);
  free(in);
  free(out);
  free(out_base);

  return 0;
}
