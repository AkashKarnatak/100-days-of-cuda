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

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define BENCHMARK_CUDA_KERNEL(name, warmup_iters, bench_iters, kernel_call)    \
  do {                                                                         \
    printf("Benchmarking %s...\n", name);                                      \
    /* warmup */                                                               \
    for (int _i = 0; _i < warmup_iters; ++_i) {                                \
      kernel_call;                                                             \
    }                                                                          \
    CUDA_CHECK(cudaDeviceSynchronize());                                       \
                                                                               \
    cudaEvent_t _start, _stop;                                                 \
    CUDA_CHECK(cudaEventCreate(&_start));                                      \
    CUDA_CHECK(cudaEventCreate(&_stop));                                       \
                                                                               \
    /* benchmark */                                                            \
    CUDA_CHECK(cudaEventRecord(_start));                                       \
    for (int _i = 0; _i < bench_iters; ++_i) {                                 \
      kernel_call;                                                             \
    }                                                                          \
    CUDA_CHECK(cudaEventRecord(_stop));                                        \
    CUDA_CHECK(cudaEventSynchronize(_stop));                                   \
                                                                               \
    float _elapsed_ms = 0.0f;                                                  \
    CUDA_CHECK(cudaEventElapsedTime(&_elapsed_ms, _start, _stop));             \
                                                                               \
    printf("Average runtime for %s: %.4f ms\n\n", name,                        \
           _elapsed_ms / bench_iters);                                         \
                                                                               \
    CUDA_CHECK(cudaEventDestroy(_start));                                      \
    CUDA_CHECK(cudaEventDestroy(_stop));                                       \
  } while (0)

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
          const size_t CM>
__global__ void matmul_tiled_2d_kernel(float *A, float *B, float *C, size_t N,
                                       size_t K, size_t M) {
  const size_t TN = BN / CN;
  const size_t TM = BM / CM;
  const size_t rowAOffset = blockIdx.y * BN;
  const size_t colBOffset = blockIdx.x * BM;
  const size_t rowCOffset = rowAOffset;
  const size_t colCOffset = colBOffset;

  size_t cnt = 0;
  size_t innerColA = threadIdx.x % BK;
  size_t innerRowA = threadIdx.x / BK;
  size_t innerColB = threadIdx.x % BM;
  size_t innerRowB = threadIdx.x / BM;
  size_t innerColC = threadIdx.x % CM;
  size_t innerRowC = threadIdx.x / CM;

  // CN * CM is the number of threads in this block
  size_t strideA = CN * CM / BK;
  size_t strideB = CN * CM / BM;

  __shared__ float A_s[BN][BK];
  __shared__ float B_s[BK][BM];

  float sums[TN * TM] = {0};
  float A_reg[TN], B_reg[TM];

  for (size_t tileOffset = 0; tileOffset < K; tileOffset += BK) {

    // load data in shared memory
    for (size_t innerRowAOffset = 0; innerRowAOffset < BN;
         innerRowAOffset += strideA) {
      if ((rowAOffset + innerRowAOffset + innerRowA) < N &&
          (tileOffset + innerColA) < K)
        A_s[innerRowAOffset + innerRowA][innerColA] =
            A[(rowAOffset + innerRowAOffset + innerRowA) * K + tileOffset +
              innerColA];
      else
        A_s[innerRowAOffset + innerRowA][innerColA] = 0.0f;
    }

    for (size_t innerRowBOffset = 0; innerRowBOffset < BK;
         innerRowBOffset += strideB) {
      if ((tileOffset + innerRowBOffset + innerRowB) < K &&
          (colBOffset + innerColB) < M)
        B_s[innerRowBOffset + innerRowB][innerColB] =
            B[(tileOffset + innerRowBOffset + innerRowB) * M + colBOffset +
              innerColB];
      else
        B_s[innerRowBOffset + innerRowB][innerColB] = 0.0f;
    }

    __syncthreads();

    // compute
    for (size_t k = 0; k < BK; ++k) {
      cnt = 0;

      // load value in registers
      for (size_t i = 0; i < TN; ++i) {
        A_reg[i] = A_s[i * CN + innerRowC][k];
      }
      for (size_t j = 0; j < TM; ++j) {
        B_reg[j] = B_s[k][j * CM + innerColC];
      }

      for (size_t i = 0; i < TN; ++i) {
        for (size_t j = 0; j < TM; ++j) {
          sums[cnt++] += A_reg[i] * B_reg[j];
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
      if ((rowCOffset + innerRowCOffset + innerRowC) < N &&
          (colCOffset + innerColCOffset + innerColC) < M) {
        C[(rowCOffset + innerRowCOffset + innerRowC) * M + colCOffset +
          innerColCOffset + innerColC] = sums[cnt];
      }
      ++cnt;
    }
  }
}

template <const size_t BN, const size_t BK, const size_t BM, const size_t CN,
          const size_t CM>
__global__ void matmul_tiled_vector_kernel(float *A, float *B, float *C,
                                           size_t N, size_t K, size_t M) {
  const size_t TN = BN / CN;
  const size_t TM = BM / CM;
  const size_t rowAOffset = blockIdx.y * BN;
  const size_t colBOffset = blockIdx.x * BM;
  const size_t rowCOffset = rowAOffset;
  const size_t colCOffset = colBOffset;

  size_t cnt = 0;
  size_t innerColA = threadIdx.x % (BK / 4);
  size_t innerRowA = threadIdx.x / (BK / 4);
  size_t innerColB = threadIdx.x % (BM / 4);
  size_t innerRowB = threadIdx.x / (BM / 4);
  size_t innerColC = threadIdx.x % CM;
  size_t innerRowC = threadIdx.x / CM;

  // CN * CM is the number of threads in this block
  size_t strideA = CN * CM / (BK / 4);
  size_t strideB = CN * CM / (BM / 4);

  __shared__ float A_s[BK][BN];
  __shared__ float B_s[BK][BM + 5];

  float sums[TN * TM] = {0};
  float A_reg[TN], B_reg[TM];

  for (size_t tileOffset = 0; tileOffset < K; tileOffset += BK) {

    // load data in shared memory
    for (size_t innerRowAOffset = 0; innerRowAOffset < BN;
         innerRowAOffset += strideA) {
      if ((rowAOffset + innerRowAOffset + innerRowA) < N &&
          (tileOffset + innerColA * 4) < K) {
        float4 tmp =
            *((float4 *)&A[(rowAOffset + innerRowAOffset + innerRowA) * K +
                           tileOffset + innerColA * 4]);
        // TODO fix bank conflicts
        A_s[innerColA * 4 + 0][innerRowAOffset + innerRowA] = tmp.x;
        A_s[innerColA * 4 + 1][innerRowAOffset + innerRowA] = tmp.y;
        A_s[innerColA * 4 + 2][innerRowAOffset + innerRowA] = tmp.z;
        A_s[innerColA * 4 + 3][innerRowAOffset + innerRowA] = tmp.w;
      } else {
        A_s[innerColA * 4 + 0][innerRowAOffset + innerRowA] = 0.0f;
        A_s[innerColA * 4 + 1][innerRowAOffset + innerRowA] = 0.0f;
        A_s[innerColA * 4 + 2][innerRowAOffset + innerRowA] = 0.0f;
        A_s[innerColA * 4 + 3][innerRowAOffset + innerRowA] = 0.0f;
      }
    }

    for (size_t innerRowBOffset = 0; innerRowBOffset < BK;
         innerRowBOffset += strideB) {
      if ((tileOffset + innerRowBOffset + innerRowB) < K &&
          (colBOffset + innerColB * 4) < M) {
        float4 tmp =
            *(float4 *)&B[(tileOffset + innerRowBOffset + innerRowB) * M +
                          colBOffset + innerColB * 4];
        B_s[innerRowBOffset + innerRowB][innerColB * 4 + 0] = tmp.x;
        B_s[innerRowBOffset + innerRowB][innerColB * 4 + 1] = tmp.y;
        B_s[innerRowBOffset + innerRowB][innerColB * 4 + 2] = tmp.z;
        B_s[innerRowBOffset + innerRowB][innerColB * 4 + 3] = tmp.w;
      } else {
        B_s[innerRowBOffset + innerRowB][innerColB * 4 + 0] = 0.0f;
        B_s[innerRowBOffset + innerRowB][innerColB * 4 + 1] = 0.0f;
        B_s[innerRowBOffset + innerRowB][innerColB * 4 + 2] = 0.0f;
        B_s[innerRowBOffset + innerRowB][innerColB * 4 + 3] = 0.0f;
      }
    }

    __syncthreads();

    // compute
    for (size_t k = 0; k < BK; ++k) {
      cnt = 0;

      // load value in registers
      for (size_t i = 0; i < TN; ++i) {
        A_reg[i] = A_s[k][i * CN + innerRowC];
      }
      for (size_t j = 0; j < TM; ++j) {
        B_reg[j] = B_s[k][j * CM + innerColC];
      }

      for (size_t i = 0; i < TN; ++i) {
        for (size_t j = 0; j < TM; ++j) {
          sums[cnt++] += A_reg[i] * B_reg[j];
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
      if ((rowCOffset + innerRowCOffset + innerRowC) < N &&
          (colCOffset + innerColCOffset + innerColC) < M) {
        C[(rowCOffset + innerRowCOffset + innerRowC) * M + colCOffset +
          innerColCOffset + innerColC] = sums[cnt];
      }
      ++cnt;
    }
  }
}

template <const size_t BN, const size_t BK, const size_t BM, const size_t TN,
          const size_t TM>
__global__ void matmul_tiled_vector_kernel2(float *A, float *B, float *C,
                                            size_t N, size_t K, size_t M) {
  const size_t rowAOffset = blockIdx.y * BN;
  const size_t colBOffset = blockIdx.x * BM;
  const size_t rowCOffset = rowAOffset;
  const size_t colCOffset = colBOffset;

  size_t cnt = 0;
  size_t innerColA = threadIdx.x % (BK / 4);
  size_t innerRowA = threadIdx.x / (BK / 4);
  size_t innerColB = threadIdx.x % (BM / 4);
  size_t innerRowB = threadIdx.x / (BM / 4);
  size_t threadCol = threadIdx.x % (BM / TM);
  size_t threadRow = threadIdx.x / (BM / TM);

  size_t numThreads = (BN * BM) / (TN * TM);
  size_t strideA = numThreads / (BK / 4);
  size_t strideB = numThreads / (BM / 4);

  __shared__ float A_s[BK][BN];
  __shared__ float B_s[BK][BM + 5];

  float sums[TN * TM] = {0};
  float A_reg[TN], B_reg[TM];

  for (size_t tileOffset = 0; tileOffset < K; tileOffset += BK) {

    // load data in shared memory
    for (size_t innerRowAOffset = 0; innerRowAOffset < BN;
         innerRowAOffset += strideA) {
      if ((rowAOffset + innerRowAOffset + innerRowA) < N &&
          (tileOffset + innerColA * 4) < K) {
        float4 tmp =
            *((float4 *)&A[(rowAOffset + innerRowAOffset + innerRowA) * K +
                           tileOffset + innerColA * 4]);
        // TODO fix bank conflicts
        A_s[innerColA * 4 + 0][innerRowAOffset + innerRowA] = tmp.x;
        A_s[innerColA * 4 + 1][innerRowAOffset + innerRowA] = tmp.y;
        A_s[innerColA * 4 + 2][innerRowAOffset + innerRowA] = tmp.z;
        A_s[innerColA * 4 + 3][innerRowAOffset + innerRowA] = tmp.w;
      } else {
        A_s[innerColA * 4 + 0][innerRowAOffset + innerRowA] = 0.0f;
        A_s[innerColA * 4 + 1][innerRowAOffset + innerRowA] = 0.0f;
        A_s[innerColA * 4 + 2][innerRowAOffset + innerRowA] = 0.0f;
        A_s[innerColA * 4 + 3][innerRowAOffset + innerRowA] = 0.0f;
      }
    }

    for (size_t innerRowBOffset = 0; innerRowBOffset < BK;
         innerRowBOffset += strideB) {
      if ((tileOffset + innerRowBOffset + innerRowB) < K &&
          (colBOffset + innerColB * 4) < M) {
        float4 tmp =
            *(float4 *)&B[(tileOffset + innerRowBOffset + innerRowB) * M +
                          colBOffset + innerColB * 4];
        B_s[innerRowBOffset + innerRowB][innerColB * 4 + 0] = tmp.x;
        B_s[innerRowBOffset + innerRowB][innerColB * 4 + 1] = tmp.y;
        B_s[innerRowBOffset + innerRowB][innerColB * 4 + 2] = tmp.z;
        B_s[innerRowBOffset + innerRowB][innerColB * 4 + 3] = tmp.w;
      } else {
        B_s[innerRowBOffset + innerRowB][innerColB * 4 + 0] = 0.0f;
        B_s[innerRowBOffset + innerRowB][innerColB * 4 + 1] = 0.0f;
        B_s[innerRowBOffset + innerRowB][innerColB * 4 + 2] = 0.0f;
        B_s[innerRowBOffset + innerRowB][innerColB * 4 + 3] = 0.0f;
      }
    }

    __syncthreads();

    // compute
    for (size_t k = 0; k < BK; ++k) {
      cnt = 0;

      // load value in registers
      for (size_t i = 0; i < TN; ++i) {
        A_reg[i] = A_s[k][threadRow * TN + i];
      }
      for (size_t j = 0; j < TM; ++j) {
        B_reg[j] = B_s[k][threadCol * TM + j];
      }

      for (size_t i = 0; i < TN; ++i) {
        for (size_t j = 0; j < TM; ++j) {
          sums[cnt++] += A_reg[i] * B_reg[j];
        }
      }
    }
    __syncthreads();
  }

  cnt = 0;
  for (size_t i = 0; i < TN; ++i) {
    for (size_t j = 0; j < TM; ++j) {
      size_t innerRowC = threadRow * TN + i;
      size_t innerColC = threadCol * TM + j;
      if ((rowCOffset + innerRowC) < N && (colCOffset + innerColC) < M) {
        C[(rowCOffset + innerRowC) * M + colCOffset + innerColC] = sums[cnt];
      }
      ++cnt;
    }
  }
}

template <const size_t BN, const size_t BK, const size_t BM, const size_t TN,
          const size_t TM, const size_t WITERN, const size_t WITERM,
          const size_t WCOL>
__global__ void matmul_tiled_warptile_kernel(float *A, float *B, float *C,
                                             size_t N, size_t K, size_t M) {
  const size_t WARPSIZE = 32;
  const size_t WSUBN = WITERN * TN;
  const size_t WSUBM = WITERM * TM;
  const size_t WN = WSUBN * 32 / WCOL;
  const size_t WM = WSUBM * WCOL;

  const size_t rowAOffset = blockIdx.y * BN;
  const size_t colBOffset = blockIdx.x * BM;
  const size_t rowCOffset = rowAOffset;
  const size_t colCOffset = colBOffset;

  size_t cnt = 0;
  size_t innerColA = threadIdx.x % (BK / 4);
  size_t innerRowA = threadIdx.x / (BK / 4);
  size_t innerColB = threadIdx.x % (BM / 4);
  size_t innerRowB = threadIdx.x / (BM / 4);

  size_t laneIdx = threadIdx.x % WARPSIZE;
  size_t warpIdx = threadIdx.x / WARPSIZE;
  size_t innerWarpColOffset = (laneIdx % WCOL) * WSUBM;
  size_t innerWarpRowOffset = (laneIdx / WCOL) * WSUBN;
  size_t warpColOffset = warpIdx % (BM / WM) * WM;
  size_t warpRowOffset = warpIdx / (BM / WM) * WN;

  size_t numThreads = (BN * BM) / (WSUBN * WSUBM);
  size_t strideA = numThreads / (BK / 4);
  size_t strideB = numThreads / (BM / 4);

  __shared__ float A_s[BK][BN];
  __shared__ float B_s[BK][BM + 5];

  float sums[WSUBN * WSUBM] = {0};
  float A_reg[WSUBN], B_reg[WSUBM];

  for (size_t tileOffset = 0; tileOffset < K; tileOffset += BK) {

    // load data in shared memory
    for (size_t innerRowAOffset = 0; innerRowAOffset < BN;
         innerRowAOffset += strideA) {
      if ((rowAOffset + innerRowAOffset + innerRowA) < N &&
          (tileOffset + innerColA * 4) < K) {
        float4 tmp =
            *((float4 *)&A[(rowAOffset + innerRowAOffset + innerRowA) * K +
                           tileOffset + innerColA * 4]);
        // TODO fix bank conflicts
        A_s[innerColA * 4 + 0][innerRowAOffset + innerRowA] = tmp.x;
        A_s[innerColA * 4 + 1][innerRowAOffset + innerRowA] = tmp.y;
        A_s[innerColA * 4 + 2][innerRowAOffset + innerRowA] = tmp.z;
        A_s[innerColA * 4 + 3][innerRowAOffset + innerRowA] = tmp.w;
      } else {
        A_s[innerColA * 4 + 0][innerRowAOffset + innerRowA] = 0.0f;
        A_s[innerColA * 4 + 1][innerRowAOffset + innerRowA] = 0.0f;
        A_s[innerColA * 4 + 2][innerRowAOffset + innerRowA] = 0.0f;
        A_s[innerColA * 4 + 3][innerRowAOffset + innerRowA] = 0.0f;
      }
    }

    for (size_t innerRowBOffset = 0; innerRowBOffset < BK;
         innerRowBOffset += strideB) {
      if ((tileOffset + innerRowBOffset + innerRowB) < K &&
          (colBOffset + innerColB * 4) < M) {
        float4 tmp =
            *(float4 *)&B[(tileOffset + innerRowBOffset + innerRowB) * M +
                          colBOffset + innerColB * 4];
        B_s[innerRowBOffset + innerRowB][innerColB * 4 + 0] = tmp.x;
        B_s[innerRowBOffset + innerRowB][innerColB * 4 + 1] = tmp.y;
        B_s[innerRowBOffset + innerRowB][innerColB * 4 + 2] = tmp.z;
        B_s[innerRowBOffset + innerRowB][innerColB * 4 + 3] = tmp.w;
      } else {
        B_s[innerRowBOffset + innerRowB][innerColB * 4 + 0] = 0.0f;
        B_s[innerRowBOffset + innerRowB][innerColB * 4 + 1] = 0.0f;
        B_s[innerRowBOffset + innerRowB][innerColB * 4 + 2] = 0.0f;
        B_s[innerRowBOffset + innerRowB][innerColB * 4 + 3] = 0.0f;
      }
    }

    __syncthreads();

    // compute
    for (size_t k = 0; k < BK; ++k) {
      cnt = 0;

      // load value in registers
      for (size_t wSubRowOffset = 0; wSubRowOffset < WSUBN;
           wSubRowOffset += TN) {
        for (size_t i = 0; i < TN; ++i) {
          A_reg[wSubRowOffset + i] =
              A_s[k][warpRowOffset + innerWarpRowOffset + wSubRowOffset + i];
        }
      }
      for (size_t wSubColOffset = 0; wSubColOffset < WSUBM;
           wSubColOffset += TM) {
        for (size_t j = 0; j < TM; ++j) {
          B_reg[wSubColOffset + j] =
              B_s[k][warpColOffset + innerWarpColOffset + wSubColOffset + j];
        }
      }

      for (size_t wSubRowOffset = 0; wSubRowOffset < WSUBN;
           wSubRowOffset += TN) {
        for (size_t wSubColOffset = 0; wSubColOffset < WSUBM;
             wSubColOffset += TM) {
          for (size_t i = 0; i < TN; ++i) {
            for (size_t j = 0; j < TM; ++j) {
              sums[cnt++] +=
                  A_reg[wSubRowOffset + i] * B_reg[wSubColOffset + j];
            }
          }
        }
      }
    }
    __syncthreads();
  }

  cnt = 0;
  for (size_t wSubRowOffset = 0; wSubRowOffset < WSUBN; wSubRowOffset += TN) {
    for (size_t wSubColOffset = 0; wSubColOffset < WSUBM; wSubColOffset += TM) {
      for (size_t i = 0; i < TN; ++i) {
        for (size_t j = 0; j < TM; ++j) {
          size_t innerRowC =
              warpRowOffset + innerWarpRowOffset + wSubRowOffset + i;
          size_t innerColC =
              warpColOffset + innerWarpColOffset + wSubColOffset + j;
          if ((rowCOffset + innerRowC) < N && (colCOffset + innerColC) < M) {
            C[(rowCOffset + innerRowC) * M + colCOffset + innerColC] =
                sums[cnt];
          }
          ++cnt;
        }
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

  CUDA_CHECK(cudaMemcpy(A_d, A, N * K * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(B_d, B, K * M * sizeof(float), cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaDeviceSynchronize());
  const size_t BLOCK_SIZE = 32;
  numThreads = dim3(BLOCK_SIZE * BLOCK_SIZE);
  numBlocks = dim3(cdiv(M, BLOCK_SIZE), cdiv(N, BLOCK_SIZE));
  matmul_naive_kernel<BLOCK_SIZE>
      <<<numBlocks, numThreads>>>(A_d, B_d, C_base_d, N, K, M);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(C_base, C_base_d, N * M * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // // naive matmul kernel
  // cudaMemset(C_d, 0, N * M * sizeof(float));
  // numThreads = dim3(BLOCK_SIZE * BLOCK_SIZE);
  // numBlocks = dim3(cdiv(M, BLOCK_SIZE), cdiv(N, BLOCK_SIZE));
  // BENCHMARK_CUDA_KERNEL("naive matmul kernel", 5, 20,
  //                       (matmul_naive_kernel<BLOCK_SIZE>
  //                        <<<numBlocks, numThreads>>>(A_d, B_d, C_d, N, K,
  //                        M)));
  // CUDA_CHECK(cudaMemcpy(C, C_d, N * M * sizeof(float),
  // cudaMemcpyDeviceToHost)); printf("Match impl: %s\n\n", allclose(C_base, C,
  // N, M) ? "true" : "false");

  // // tiled matmul kernel
  // cudaMemset(C_d, 0, N * M * sizeof(float));
  // numThreads = dim3(BLOCK_SIZE * BLOCK_SIZE);
  // numBlocks = dim3(cdiv(M, BLOCK_SIZE), cdiv(N, BLOCK_SIZE));
  // BENCHMARK_CUDA_KERNEL("tiled matmul kernel", 5, 20,
  //                       (matmul_tiled_kernel<BLOCK_SIZE>
  //                        <<<numBlocks, numThreads>>>(A_d, B_d, C_d, N, K,
  //                        M)));
  // CUDA_CHECK(cudaMemcpy(C, C_d, N * M * sizeof(float),
  // cudaMemcpyDeviceToHost)); printf("Match impl: %s\n\n", allclose(C_base, C,
  // N, M) ? "true" : "false");

  // // blocktiled matmul kernel
  // cudaMemset(C_d, 0, N * M * sizeof(float));
  // const size_t CN = 8;
  // const size_t CM = 8;
  // const size_t BK = 8;
  // const size_t BN = 64;
  // const size_t BM = 64;
  // assert(BN * BK >= CN * CM &&
  //        BM * BK >= CN * CM); // number of threads must be less than
  // numThreads = dim3(CN * CM);
  // numBlocks = dim3(cdiv(M, BM), cdiv(N, BN));
  // BENCHMARK_CUDA_KERNEL("block tiled matmul kernel", 5, 20,
  //                       (matmul_tiled_2d_kernel<BN, BK, BM, CN, CM>
  //                        <<<numBlocks, numThreads>>>(A_d, B_d, C_d, N, K,
  //                        M)));
  // CUDA_CHECK(cudaMemcpy(C, C_d, N * M * sizeof(float),
  // cudaMemcpyDeviceToHost)); printf("Match impl: %s\n\n", allclose(C_base, C,
  // N, M) ? "true" : "false");

  // // vectorized block tiled matmul kernel
  // CUDA_CHECK(cudaDeviceSynchronize());
  // cudaMemset(C_d, 0, N * M * sizeof(float));
  // const size_t CN = 8;
  // const size_t CM = 8;
  // const size_t BK = 8;
  // const size_t BN = 64;
  // const size_t BM = 64;
  // assert(BN * BK >= 4 * CN * CM &&
  //        BM * BK >= 4 * CN * CM); // number of threads must be less than
  // numThreads = dim3(CN * CM);
  // numBlocks = dim3(cdiv(M, BM), cdiv(N, BN));
  // BENCHMARK_CUDA_KERNEL("vectorized block tiled matmul kernel", 5, 20,
  //                       (matmul_tiled_vector_kernel<BN, BK, BM, CN, CM>
  //                        <<<numBlocks, numThreads>>>(A_d, B_d, C_d, N, K,
  //                        M)));
  // CUDA_CHECK(cudaMemcpy(C, C_d, N * M * sizeof(float),
  // cudaMemcpyDeviceToHost)); printf("Match impl: %s\n\n", allclose(C_base, C,
  // N, M) ? "true" : "false");

  // // vectorized block tiled matmul kernel
  // CUDA_CHECK(cudaDeviceSynchronize());
  // cudaMemset(C_d, 0, N * M * sizeof(float));
  // const size_t TN = 8;
  // const size_t TM = 4;
  // const size_t BK = 16;
  // const size_t BN = 128;
  // const size_t BM = 64;
  // numThreads = dim3((BN * BM) / (TN * TM));
  // numBlocks = dim3(cdiv(M, BM), cdiv(N, BN));
  // BENCHMARK_CUDA_KERNEL("vectorized block tiled matmul kernel", 5, 20,
  //                       (matmul_tiled_vector_kernel2<BN, BK, BM, TN, TM>
  //                        <<<numBlocks, numThreads>>>(A_d, B_d, C_d, N, K,
  //                        M)));
  // CUDA_CHECK(cudaMemcpy(C, C_d, N * M * sizeof(float),
  // cudaMemcpyDeviceToHost)); printf("Match impl: %s\n\n", allclose(C_base, C,
  // N, M) ? "true" : "false");

  // vectorized block tiled matmul kernel
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaMemset(C_d, 0, N * M * sizeof(float));
  const size_t TN = 2;
  const size_t TM = 2;
  const size_t BK = 8;
  const size_t BN = 32;
  const size_t BM = 32;
  const size_t WITERN = 2;
  const size_t WITERM = 2;
  const size_t WCOL = 4;
  assert(BN * BM > 32 * TN * TM * WITERN * WITERM); // have atleast 1 warp
  numThreads = dim3((BN * BM) / (TN * TM * WITERN * WITERM));
  numBlocks = dim3(cdiv(M, BM), cdiv(N, BN));
  BENCHMARK_CUDA_KERNEL(
      "vectorized block tiled matmul kernel", 5, 20,
      (matmul_tiled_warptile_kernel<BN, BK, BM, TN, TM, WITERN, WITERN, WCOL>
       <<<numBlocks, numThreads>>>(A_d, B_d, C_d, N, K, M)));
  CUDA_CHECK(cudaMemcpy(C, C_d, N * M * sizeof(float), cudaMemcpyDeviceToHost));
  printf("Match impl: %s\n\n", allclose(C_base, C, N, M) ? "true" : "false");

  // print_mat(C, N, M);
  // print_mat(C_base, N, M);

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
