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

// CM = BM, because there is no advantage of having CM less than BM
template <const size_t BN, const size_t BM, const size_t CN>
__global__ void transpose_shared_kernel(float *in, float *out, size_t N,
                                        size_t M) {
  size_t rowOffsetIn = blockIdx.y * BN;
  size_t colOffsetIn = blockIdx.x * BM;
  size_t innerColIn = threadIdx.x % BM;
  size_t innerRowIn = threadIdx.x / BM;

  __shared__ float sram[BN][BM + 1];

  // coalasced read from global and coalasced store to sram
  for (size_t innerRowOffset = 0; innerRowOffset < BN; innerRowOffset += CN) {
    sram[innerRowOffset + innerRowIn][innerColIn] =
        in[(rowOffsetIn + innerRowOffset + innerRowIn) * M +
           (colOffsetIn + innerColIn)];
  }

  __syncthreads();

  size_t rowOffsetOut = colOffsetIn;
  size_t colOffsetOut = rowOffsetIn;
  size_t innerColOut = threadIdx.x % BN;
  size_t innerRowOut = threadIdx.x / BN;
  size_t stride =
      (CN * BM) / BN; // CN * BM is total number of threads in this block

  // read from sram without bank conflict and coalasced store in global
  for (size_t innerRowOffset = 0; innerRowOffset < BM;
       innerRowOffset += stride) {
    out[(rowOffsetOut + innerRowOffset + innerRowOut) * N +
        (colOffsetOut + innerColOut)] =
        sram[innerColOut][innerRowOffset + innerRowOut];
  }
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

void transpose_cpu(float *in, float *out, size_t N, size_t M) {
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < M; ++j) {
      out[j * N + i] = in[i * M + j];
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

  transpose_cpu(in, out_base, N, M);

  CUDA_CHECK(
      cudaMemcpy(in_d, in, N * M * sizeof(float), cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaDeviceSynchronize());
  cudaMemset(out_d, 0, N * M * sizeof(float));
  start_timer(&t);
  const size_t BN = 32;
  const size_t BM = 32;
  const size_t CN = 8;
  numThreads = dim3(BM * CN);
  numBlocks = dim3(cdiv(M, BM), cdiv(N, BN));
  transpose_shared_kernel<BN, BM, CN>
      <<<numBlocks, numThreads>>>(in_d, out_d, N, M);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  stop_timer(&t);
  CUDA_CHECK(
      cudaMemcpy(out, out_d, N * M * sizeof(float), cudaMemcpyDeviceToHost));
  printf("Transpose kernel with shared mem time: %f\n", time_diff(&t));
  printf("Match impl: %s\n\n",
         allclose(out_base, out, N, M) ? "true" : "false");

  cudaFree(in_d);
  cudaFree(out_d);
  free(in);
  free(out);
  free(out_base);

  return 0;
}
