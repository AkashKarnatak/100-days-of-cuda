#include <cstdint>
#include <stdio.h>

#define CONV_RADIUS 2
#define CONV_DIM ((CONV_RADIUS * 2) + 1)
#define IN_TILE_DIM 32
#define OUT_TILE_DIM (IN_TILE_DIM - 2 * CONV_RADIUS)

__constant__ float filter_c[CONV_DIM][CONV_DIM];

__host__ __device__ size_t cdiv(size_t a, size_t b) { return (a + b - 1) / b; }

__global__ void naive_conv2d_kernel(float *in, float *out, size_t N, size_t M) {
  size_t outRow = blockIdx.y * blockDim.y + threadIdx.y;
  size_t outCol = blockIdx.x * blockDim.x + threadIdx.x;

  if (outRow >= N || outCol >= M)
    return;

  float sum = 0;
  for (size_t filterRow = 0; filterRow < CONV_DIM; ++filterRow) {
    for (size_t filterCol = 0; filterCol < CONV_DIM; ++filterCol) {
      int32_t inRow = outRow - CONV_RADIUS + filterRow;
      int32_t inCol = outCol - CONV_RADIUS + filterCol;
      if (inRow >= 0 && inRow < (int32_t)N && inCol >= 0 &&
          inCol < (int32_t)M) {
        sum += in[inRow * M + inCol] * filter_c[filterRow][filterCol];
      }
    }
  }
  out[outRow * M + outCol] = sum;
}

__global__ void tiled_conv2d_kernel(float *in, float *out, size_t N, size_t M) {
  int32_t outRow = blockIdx.y * OUT_TILE_DIM + threadIdx.y;
  int32_t outCol = blockIdx.x * OUT_TILE_DIM + threadIdx.x;
  int32_t inRow = outRow - CONV_RADIUS;
  int32_t inCol = outCol - CONV_RADIUS;

  __shared__ float in_s[IN_TILE_DIM][IN_TILE_DIM];

  if (inRow >= 0 && inRow < (int32_t)N && inCol >= 0 && inCol < (int32_t)M) {
    in_s[threadIdx.y][threadIdx.x] = in[inRow * M + inCol];
  } else {
    in_s[threadIdx.y][threadIdx.x] = 0; // ghost cell
  }

  __syncthreads();

  if (threadIdx.y >= OUT_TILE_DIM || threadIdx.x >= OUT_TILE_DIM)
    return;

  float sum = 0;
  for (size_t filterRow = 0; filterRow < CONV_DIM; ++filterRow) {
    for (size_t filterCol = 0; filterCol < CONV_DIM; ++filterCol) {
      sum += in_s[threadIdx.y + filterRow][threadIdx.x + filterCol] *
             filter_c[filterRow][filterCol];
    }
  }

  if (outRow < N && outCol < M)
    out[outRow * M + outCol] = sum;
}

__global__ void tiled_with_cache_conv2d_kernel(float *in, float *out, size_t N,
                                               size_t M) {
  size_t outRow = blockIdx.y * OUT_TILE_DIM + threadIdx.y;
  size_t outCol = blockIdx.x * OUT_TILE_DIM + threadIdx.x;
  size_t outRowStart = blockIdx.y * OUT_TILE_DIM;
  size_t outColStart = blockIdx.x * OUT_TILE_DIM;
  size_t outRowEnd = (blockIdx.y + 1) * OUT_TILE_DIM;
  size_t outColEnd = (blockIdx.x + 1) * OUT_TILE_DIM;

  __shared__ float in_s[OUT_TILE_DIM][OUT_TILE_DIM];

  if (outRow >= N || outCol >= M)
    return;

  in_s[threadIdx.y][threadIdx.x] = in[outRow * M + outCol];

  __syncthreads();

  float sum = 0;
  for (size_t filterRow = 0; filterRow < CONV_DIM; ++filterRow) {
    for (size_t filterCol = 0; filterCol < CONV_DIM; ++filterCol) {
      int32_t inRow = outRow - CONV_RADIUS + filterRow;
      int32_t inCol = outCol - CONV_RADIUS + filterCol;
      if (inRow >= 0 && inRow < (int32_t)N && inCol >= 0 &&
          inCol < (int32_t)M) {
        if (inRow >= (int32_t)outRowStart && inRow < (int32_t)outRowEnd &&
            inCol >= (int32_t)outColStart && inCol < (int32_t)outColEnd) {
          sum += in_s[inRow - outRowStart][inCol - outColStart] *
                 filter_c[filterRow][filterCol];
        } else {
          sum += in[inRow * M + inCol] * filter_c[filterRow][filterCol];
        }
      }
    }
  }

  out[outRow * M + outCol] = sum;
}

void naive_conv2d_gpu(float *in, float *filter, float *out, size_t N,
                      size_t M) {
  float *in_d, *out_d;

  cudaMalloc(&in_d, (N * M) * sizeof(float));
  cudaMalloc(&out_d, (N * M) * sizeof(float));

  cudaMemcpy(in_d, in, (N * M) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(filter_c, filter, (CONV_DIM * CONV_DIM) * sizeof(float));

  dim3 numThreads(32, 32);
  dim3 numBlocks(cdiv(M, numThreads.x), cdiv(N, numThreads.y));
  naive_conv2d_kernel<<<numBlocks, numThreads>>>(in_d, out_d, N, M);

  cudaMemcpy(out, out_d, (N * M) * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(in_d);
  cudaFree(out_d);
}

void tiled_conv2d_gpu(float *in, float *filter, float *out, size_t N,
                      size_t M) {
  float *in_d, *out_d;

  cudaMalloc(&in_d, (N * M) * sizeof(float));
  cudaMalloc(&out_d, (N * M) * sizeof(float));

  cudaMemcpy(in_d, in, (N * M) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(filter_c, filter, (CONV_DIM * CONV_DIM) * sizeof(float));

  dim3 numThreads(IN_TILE_DIM, IN_TILE_DIM);
  dim3 numBlocks(cdiv(M, OUT_TILE_DIM), cdiv(N, OUT_TILE_DIM));
  tiled_conv2d_kernel<<<numBlocks, numThreads>>>(in_d, out_d, N, M);

  cudaMemcpy(out, out_d, (N * M) * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(in_d);
  cudaFree(out_d);
}

void tiled_with_cache_conv2d_gpu(float *in, float *filter, float *out, size_t N,
                                 size_t M) {
  float *in_d, *out_d;

  cudaMalloc(&in_d, (N * M) * sizeof(float));
  cudaMalloc(&out_d, (N * M) * sizeof(float));

  cudaMemcpy(in_d, in, (N * M) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(filter_c, filter, (CONV_DIM * CONV_DIM) * sizeof(float));

  dim3 numThreads(OUT_TILE_DIM, OUT_TILE_DIM);
  dim3 numBlocks(cdiv(M, OUT_TILE_DIM), cdiv(N, OUT_TILE_DIM));
  tiled_with_cache_conv2d_kernel<<<numBlocks, numThreads>>>(in_d, out_d, N, M);

  cudaMemcpy(out, out_d, (N * M) * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(in_d);
  cudaFree(out_d);
}

void conv2d_cpu(float *in, float *filter, float *out, size_t N, size_t M) {
  for (size_t outRow = 0; outRow < N; ++outRow) {
    for (size_t outCol = 0; outCol < M; ++outCol) {
      float sum = 0;
      for (size_t filterRow = 0; filterRow < CONV_DIM; ++filterRow) {
        for (size_t filterCol = 0; filterCol < CONV_DIM; ++filterCol) {
          int32_t inRow = (int32_t)outRow - CONV_RADIUS + (int32_t)filterRow;
          int32_t inCol = (int32_t)outCol - CONV_RADIUS + (int32_t)filterCol;
          if (inRow >= 0 && inRow < (int32_t)N && inCol >= 0 &&
              inCol < (int32_t)M) {
            sum += filter[filterRow * CONV_DIM + filterCol] *
                   in[inRow * M + inCol];
          }
        }
      }
      out[outRow * M + outCol] = sum;
    }
  }
}

bool allclose(float *A, float *B, size_t N, size_t M) {
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < M; ++j) {
      if (abs(A[i * M + j] - B[i * M + j]) > 1e-3) {
        printf("Mismatch at (%lu, %lu) where A = %f and B = %f\n", i, j,
               A[i * M + j], B[i * M + j]);
        return false;
      }
    }
  }
  return true;
}

int main() {
  size_t N, M;
  float *in, *filter, *out_cpu, *out_gpu_1, *out_gpu_2, *out_gpu_3;

  N = 1537, M = 1479;

  in = (float *)malloc((N * M) * sizeof(float));
  filter = (float *)malloc((CONV_DIM * CONV_DIM) * sizeof(float));
  out_cpu = (float *)malloc((N * M) * sizeof(float));
  out_gpu_1 = (float *)malloc((N * M) * sizeof(float));
  out_gpu_2 = (float *)malloc((N * M) * sizeof(float));
  out_gpu_3 = (float *)malloc((N * M) * sizeof(float));

  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < M; ++j) {
      in[i * M + j] = (float)rand() / RAND_MAX;
    }
  }
  for (size_t i = 0; i < CONV_DIM; ++i) {
    for (size_t j = 0; j < CONV_DIM; ++j) {
      filter[i * CONV_DIM + j] = (float)rand() / RAND_MAX;
    }
  }

  conv2d_cpu(in, filter, out_cpu, N, M);
  naive_conv2d_gpu(in, filter, out_gpu_1, N, M);
  tiled_conv2d_gpu(in, filter, out_gpu_2, N, M);
  tiled_with_cache_conv2d_gpu(in, filter, out_gpu_3, N, M);

  printf("Naive impl match: %s\n",
         allclose(out_cpu, out_gpu_1, N, M) ? "true" : "false");
  printf("Tiled impl match: %s\n",
         allclose(out_cpu, out_gpu_2, N, M) ? "true" : "false");
  printf("Tiled with cache match: %s\n",
         allclose(out_cpu, out_gpu_3, N, M) ? "true" : "false");

  // for (size_t i = 0; i < N; ++i) {
  //   for (size_t j = 0; j < M; ++j) {
  //     printf(" %.03f", out_gpu[i * M + j]);
  //   }
  //   printf("\n");
  // }

  free(in);
  free(filter);
  free(out_cpu);
  free(out_gpu_1);
  free(out_gpu_2);
  free(out_gpu_3);

  return 0;
}
