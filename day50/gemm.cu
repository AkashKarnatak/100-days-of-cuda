#include <cublas_v2.h>
#include <stdio.h>
#include <time.h>

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

template <const size_t BLOCK_SIZE>
__global__ void matmul_naive_kernel(float *A, float *B, float *C, size_t N,
                                    size_t K, size_t M) {
  size_t ty = threadIdx.x / BLOCK_SIZE;
  size_t tx = threadIdx.x % BLOCK_SIZE;
  size_t row = blockIdx.y * BLOCK_SIZE + ty;
  size_t col = blockIdx.x * BLOCK_SIZE + tx;

  if (row >= N && col >= M)
    return;

  float sum = 0.0f;
  for (size_t k = 0; k < K; ++k) {
    sum += A[row * K + k] * B[k * M + col];
  }
  C[row * M + col] = sum;
}

template <const size_t BLOCK_SIZE>
__global__ void matmul_tiled_kernel(float *A, float *B, float *C, size_t N,
                                    size_t K, size_t M) {
  size_t innerRow = threadIdx.x / BLOCK_SIZE;
  size_t innerCol = threadIdx.x % BLOCK_SIZE;
  size_t row = blockIdx.y * BLOCK_SIZE + innerRow;
  size_t col = blockIdx.x * BLOCK_SIZE + innerCol;

  if (row >= N && col >= M)
    return;

  __shared__ float A_s[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float B_s[BLOCK_SIZE][BLOCK_SIZE];

  float sum = 0.0f;
  for (size_t tileOffset = 0; tileOffset < K; tileOffset += BLOCK_SIZE) {
    // load data in shared memory
    if (row < N && (tileOffset + innerCol) < M)
      A_s[innerRow][innerCol] = A[row * K + tileOffset + innerCol];
    else
      A_s[innerRow][innerCol] = 0.0f;

    if ((tileOffset + innerRow) < N && col < M)
      B_s[innerRow][innerCol] = B[(tileOffset + innerRow) * M + col];
    else
      B_s[innerRow][innerCol] = 0.0f;

    __syncthreads();

    // compute
    for (size_t k = 0; k < BLOCK_SIZE; ++k) {
      sum += A_s[innerRow][k] * B_s[k][innerCol];
    }

    __syncthreads();
  }

  if (row < N && col < M)
    C[row * M + col] = sum;
}

template <const size_t BN, const size_t BK, const size_t BM, const size_t CN,
          const size_t CM, const size_t TN, const size_t TM>
__global__ void matmul_tiled_2d_kernel(float *A, float *B, float *C, size_t N,
                                       size_t K, size_t M) {
  const size_t rowAOffset = blockIdx.y * BN;
  const size_t colBOffset = blockIdx.x * BM;
  const size_t rowCOffset = rowAOffset;
  const size_t colCOffset = colBOffset;

  size_t innerColA = threadIdx.x % BK;
  size_t innerRowA = threadIdx.x / BK;
  size_t innerColB = threadIdx.x % BM;
  size_t innerRowB = threadIdx.x / BM;
  size_t innerColC = threadIdx.x % warpSize;
  size_t innerRowC = threadIdx.x / warpSize;
  size_t cnt = 0;

  __shared__ float A_s[BN][BK];
  __shared__ float B_s[BK][BM];

  float sums[TN * TN] = {0};
  for (size_t tileOffset = 0; tileOffset < K; tileOffset += BK) {

    // load data in shared memory
    if ((rowAOffset + innerRowA) < N && (tileOffset + innerColA) < M)
      A_s[innerRowA][innerColA] =
          A[(rowAOffset + innerRowA) * K + tileOffset + innerColA];
    else
      A_s[innerRowA][innerColA] = 0.0f;

    if ((tileOffset + innerRowB) < N && (colBOffset + innerColB) < M)
      B_s[innerRowB][innerColB] =
          B[(tileOffset + innerRowB) * M + colBOffset + innerColB];
    else
      B_s[innerRowB][innerColB] = 0.0f;

    __syncthreads();

    // compute
    for (size_t k = 0; k < BK; ++k) {
      cnt = 0;
      for (size_t innerRowCOffset = 0; innerRowCOffset < BN;
           innerRowCOffset += CN) {
        float A_s_tmp = A_s[innerRowCOffset + innerRowC][k];
        for (size_t innerColCOffset = 0; innerColCOffset < BM;
             innerColCOffset += CM) {
          sums[cnt++] += A_s_tmp * B_s[k][innerColCOffset + innerColC];
        }
      }
    }
    __syncthreads();
  }

  cnt = 0;
  for (size_t innerRowCOffset = 0; innerRowCOffset < BN;
       innerRowCOffset += CN) {
    for (size_t innerColCOffset = 0; innerColCOffset < BM;
         innerColCOffset += CM) {
      if ((innerRowCOffset + innerRowC) < N &&
          (innerColCOffset + innerColC) < M) {
        C[(rowCOffset + innerRowCOffset + innerRowC) * M + colCOffset +
          innerColCOffset + innerColC] = sums[cnt];
        ++cnt;
      }
    }
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

int main() {
  size_t N, K, M;
  float *A, *B, *C_base, *C;
  float *A_d, *B_d, *C_base_d, *C_d;
  dim3 numThreads, numBlocks;
  cudaError_t err;

  N = K = M = 4096;

  A = (float *)malloc(N * K * sizeof(float));
  B = (float *)malloc(K * M * sizeof(float));
  C_base = (float *)malloc(N * M * sizeof(float));
  C = (float *)malloc(N * M * sizeof(float));

  cudaMalloc(&A_d, N * K * sizeof(float));
  cudaMalloc(&B_d, K * M * sizeof(float));
  cudaMalloc(&C_base_d, N * M * sizeof(float));
  cudaMalloc(&C_d, N * M * sizeof(float));

  for (size_t i = 0; i < N * K; ++i) {
    A[i] = (float)rand() / RAND_MAX;
  }

  for (size_t i = 0; i < K * M; ++i) {
    B[i] = (float)rand() / RAND_MAX;
  }

  cudaMemcpy(A_d, A, N * K * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B, K * M * sizeof(float), cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();
  const size_t BLOCK_SIZE = 32;
  numThreads = dim3(BLOCK_SIZE * BLOCK_SIZE);
  numBlocks = dim3(cdiv(M, BLOCK_SIZE), cdiv(N, BLOCK_SIZE));
  matmul_naive_kernel<BLOCK_SIZE>
      <<<numBlocks, numThreads>>>(A_d, B_d, C_base_d, N, K, M);
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    exit(1);
  }
  cudaMemcpy(C_base, C_base_d, N * M * sizeof(float), cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();
  start_timer(&t);
  numThreads = dim3(BLOCK_SIZE * BLOCK_SIZE);
  numBlocks = dim3(cdiv(M, BLOCK_SIZE), cdiv(N, BLOCK_SIZE));
  matmul_naive_kernel<BLOCK_SIZE>
      <<<numBlocks, numThreads>>>(A_d, B_d, C_d, N, K, M);
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    exit(1);
  }
  stop_timer(&t);
  cudaMemcpy(C, C_d, N * M * sizeof(float), cudaMemcpyDeviceToHost);
  printf("Naive matmul kernel time: %f\n", time_diff(&t));
  printf("Match impl: %s\n\n", allclose(C_base, C, N, M) ? "true" : "false");

  cudaDeviceSynchronize();
  start_timer(&t);
  numThreads = dim3(BLOCK_SIZE * BLOCK_SIZE);
  numBlocks = dim3(cdiv(M, BLOCK_SIZE), cdiv(N, BLOCK_SIZE));
  matmul_tiled_kernel<BLOCK_SIZE>
      <<<numBlocks, numThreads>>>(A_d, B_d, C_d, N, K, M);
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    exit(1);
  }
  stop_timer(&t);
  cudaMemcpy(C, C_d, N * M * sizeof(float), cudaMemcpyDeviceToHost);
  printf("Tiled matmul kernel time: %f\n", time_diff(&t));
  printf("Match impl: %s\n\n", allclose(C_base, C, N, M) ? "true" : "false");

  cudaDeviceSynchronize();
  const size_t CN = 32;
  const size_t CM = 32;
  const size_t BK = 8;
  const size_t BN = CN * CM / BK;
  const size_t BM = CN * CM / BK;
  const size_t TN = BN / CN;
  const size_t TM = BM / CM;
  start_timer(&t);
  numThreads = dim3(CN * CM);
  numBlocks = dim3(cdiv(M, BN), cdiv(N, BM));
  matmul_tiled_2d_kernel<BN, BK, BM, CN, CM, TN, TM>
      <<<numBlocks, numThreads>>>(A_d, B_d, C_d, N, K, M);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    exit(1);
  }
  cudaDeviceSynchronize();
  stop_timer(&t);
  cudaMemcpy(C, C_d, N * M * sizeof(float), cudaMemcpyDeviceToHost);
  printf("Tiled 2d matmul kernel time: %f\n", time_diff(&t));
  printf("Match impl: %s\n\n", allclose(C_base, C, N, M) ? "true" : "false");

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_base_d);
  cudaFree(C_d);
  free(A);
  free(B);
  free(C_base);
  free(C);

  return 0;
}
