#include <cstdint>
#include <stdio.h>
#include <time.h>

#define MASK_DIM 5
#define MASK_RADIUS 2

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
__constant__ float mask_c[MASK_DIM][MASK_DIM];

__host__ __device__ size_t cdiv(size_t a, size_t b) { return (a + b - 1) / b; }

__global__ void conv2d_kernel(float *in_data, float *out_data, size_t N,
                              size_t M) {
  size_t outRow = blockDim.y * blockIdx.y + threadIdx.y;
  size_t outCol = blockDim.x * blockIdx.y + threadIdx.x;
  if (outRow >= N || outCol >= M)
    return;

  float sum = 0.0f;
  for (size_t maskRow = 0; maskRow < MASK_DIM; ++maskRow) {
    for (size_t maskCol = 0; maskCol < MASK_DIM; ++maskCol) {
      int32_t inRow = outRow - MASK_RADIUS + maskRow;
      int32_t inCol = outCol - MASK_RADIUS + maskRow;
      if (inRow >= 0 && inRow < (int32_t)N && inCol >= 0 && inCol < (int32_t)M) {
        sum += mask_c[maskRow][maskCol] * in_data[inRow * M + inCol];
      }
    }
  }
  out_data[outRow * M + outCol] = sum;
}

void conv2d_gpu(float *in_data, float *out_data, float *mask, size_t N,
                size_t M) {
  float *in_data_d, *out_data_d;

  cudaMalloc(&in_data_d, N * M * sizeof(float));
  cudaMalloc(&out_data_d, N * M * sizeof(float));

  cudaMemcpy(in_data_d, in_data, N * M * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(mask_c, mask, MASK_DIM * MASK_DIM * sizeof(float));

  dim3 numThreads(32, 32);
  dim3 numBlocks(cdiv(M, numThreads.x), cdiv(N, numThreads.y));
  start_timer(&t);
  conv2d_kernel<<<numBlocks, numThreads>>>(in_data_d, out_data_d, N, M);
  cudaDeviceSynchronize();
  stop_timer(&t);
  printf("GPU time: %f\n", time_diff(&t));

  cudaMemcpy(out_data, out_data_d, N * M * sizeof(float),
             cudaMemcpyDeviceToHost);

  cudaFree(in_data_d);
  cudaFree(out_data_d);
}

void conv2d_cpu(float *in_data, float *out_data, float *mask, size_t N,
                size_t M) {
  start_timer(&t);
  for (size_t outRow = 0; outRow < N; ++outRow) {
    for (size_t outCol = 0; outCol < M; ++outCol) {
      float sum = 0.0f;
      for (size_t maskRow = 0; maskRow < MASK_DIM; ++maskRow) {
        for (size_t maskCol = 0; maskCol < MASK_DIM; ++maskCol) {
          int32_t inRow = outRow - MASK_RADIUS + maskRow;
          int32_t inCol = outCol - MASK_RADIUS + maskCol;
          if (inRow >= 0 && inRow < (int32_t)N && inCol >= 0 && inCol < (int32_t)M)
            sum += in_data[inRow * M + inCol] * mask[maskRow * M + maskCol];
        }
      }
      out_data[outRow * M + outCol] = sum;
    }
  }
  stop_timer(&t);
  printf("CPU time: %f\n", time_diff(&t));
}

bool allclose(float *A, float *B, size_t N, size_t M) {
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < M; ++j) {
      if (abs(A[i * M + j] - B[i * M + j]) > 1e4) {
        printf("Mismatch at (%lu, %lu), A = %f and B = %f", i, j, A[i * M + j],
               B[i * M + j]);
        return false;
      }
    }
  }
  return true;
}

int main() {
  cudaDeviceSynchronize();

  float *in_data, *out_data_cpu, *out_data_gpu, *mask;
  size_t N, M;

  N = 2048, M = 2048;
  in_data = (float *)malloc(N * M * sizeof(float));
  out_data_cpu = (float *)malloc(N * M * sizeof(float));
  out_data_gpu = (float *)malloc(N * M * sizeof(float));
  mask = (float *)malloc(MASK_DIM * MASK_DIM * sizeof(float));

  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < M; ++j) {
      in_data[i * M + j] = (float)rand() / RAND_MAX;
    }
  }

  for (size_t i = 0; i < MASK_DIM; ++i) {
    for (size_t j = 0; j < MASK_DIM; ++j) {
      mask[i * MASK_DIM + j] = (float)rand() / RAND_MAX;
    }
  }

  conv2d_cpu(in_data, out_data_cpu, mask, N, M);

  conv2d_gpu(in_data, out_data_gpu, mask, N, M);

  printf("GPU and CPU impl match: %s",
         allclose(out_data_cpu, out_data_gpu, N, M) ? "true" : "false");

  free(in_data);
  free(out_data_cpu);
  free(out_data_gpu);

  return 0;
}
