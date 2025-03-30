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

#define BLOCK_DIM 32

__host__ __device__ uint32_t cdiv(uint32_t a, uint32_t b) {
  return (a + b - 1) / b;
}

__global__ void tiled_matmul_kernel(float *A, float *B, float *C, uint32_t M,
                                    uint32_t N, uint32_t P) {
  uint32_t row = blockDim.y * blockIdx.y + threadIdx.y;
  uint32_t col = blockDim.x * blockIdx.x + threadIdx.x;

  __shared__ float A_s[BLOCK_DIM][BLOCK_DIM];
  __shared__ float B_s[BLOCK_DIM][BLOCK_DIM];

  float sum = 0.0f;
  for (uint32_t block = 0; block < cdiv(N, BLOCK_DIM); ++block) {
    if (row < M && block * BLOCK_DIM + threadIdx.x < N)
      A_s[threadIdx.y][threadIdx.x] =
          A[row * N + block * BLOCK_DIM + threadIdx.x];
    else
      A_s[threadIdx.y][threadIdx.x] = 0.0f;
    if (block * BLOCK_DIM + threadIdx.y < N && col < P)
      B_s[threadIdx.y][threadIdx.x] =
          B[(block * BLOCK_DIM + threadIdx.y) * P + col];
    else
      B_s[threadIdx.y][threadIdx.x] = 0.0f;

    __syncthreads();

    if (row < M && col < P) {
      for (uint32_t i = 0; i < BLOCK_DIM; ++i) {
        sum += A_s[threadIdx.y][i] * B_s[i][threadIdx.x];
      }
    }

    __syncthreads();
  }
  if (row < M && col < P)
    C[row * P + col] = sum;
}

__global__ void matmul_kernel(float *A, float *B, float *C, uint32_t M,
                              uint32_t N, uint32_t P) {
  uint32_t row = blockDim.y * blockIdx.y + threadIdx.y;
  uint32_t col = blockDim.x * blockIdx.x + threadIdx.x;
  if (row >= M || col >= P)
    return;

  float sum = 0.0f;
  for (uint32_t i = 0; i < N; ++i) {
    sum += A[row * N + i] * B[i * P + col];
  }
  C[row * P + col] = sum;
}

void tiled_matmul_gpu(float *A, float *B, float *C, uint32_t M, uint32_t N,
                      uint32_t P) {
  float *A_d, *B_d, *C_d;

  // allocate memory on GPU
  cudaMalloc(&A_d, (M * N) * sizeof(float));
  cudaMalloc(&B_d, (N * P) * sizeof(float));
  cudaMalloc(&C_d, (M * P) * sizeof(float));

  // copy data from CPU to GPU
  cudaMemcpy(A_d, A, (M * N) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B, (N * P) * sizeof(float), cudaMemcpyHostToDevice);

  // perform computation
  dim3 numThreads(BLOCK_DIM, BLOCK_DIM);
  dim3 numBlocks(cdiv(P, numThreads.x), cdiv(M, numThreads.y));
  start_timer(&t);
  tiled_matmul_kernel<<<numBlocks, numThreads>>>(A_d, B_d, C_d, M, N, P);
  cudaDeviceSynchronize();
  stop_timer(&t);
  printf("GPU time(tiled): %f\n", time_diff(&t));

  // copy data from GPU to CPU
  cudaMemcpy(C, C_d, (M * P) * sizeof(float), cudaMemcpyDeviceToHost);

  // free memory
  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
}

void matmul_gpu(float *A, float *B, float *C, uint32_t M, uint32_t N,
                uint32_t P) {
  float *A_d, *B_d, *C_d;

  // allocate memory on GPU
  cudaMalloc(&A_d, (M * N) * sizeof(float));
  cudaMalloc(&B_d, (N * P) * sizeof(float));
  cudaMalloc(&C_d, (M * P) * sizeof(float));

  // copy data from CPU to GPU
  cudaMemcpy(A_d, A, (M * N) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B, (N * P) * sizeof(float), cudaMemcpyHostToDevice);

  // perform computation
  dim3 numThreads(BLOCK_DIM, BLOCK_DIM);
  dim3 numBlocks(cdiv(P, numThreads.x), cdiv(M, numThreads.y));
  start_timer(&t);
  matmul_kernel<<<numBlocks, numThreads>>>(A_d, B_d, C_d, M, N, P);
  cudaDeviceSynchronize();
  stop_timer(&t);
  printf("GPU time: %f\n", time_diff(&t));

  // copy data from GPU to CPU
  cudaMemcpy(C, C_d, (M * P) * sizeof(float), cudaMemcpyDeviceToHost);

  // free memory
  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
}

void matmul_cpu(float *A, float *B, float *C, uint32_t M, uint32_t N,
                uint32_t P) {
  start_timer(&t);
  for (uint32_t row = 0; row < M; ++row) {
    for (uint32_t col = 0; col < P; ++col) {
      float sum = 0.0f;
      for (uint32_t i = 0; i < N; ++i) {
        sum += A[row * N + i] * B[i * P + col];
      }
      C[row * P + col] = sum;
    }
  }
  stop_timer(&t);
  printf("CPU time: %f\n", time_diff(&t));
}

void tiled_matmul_cpu(float *A, float *B, float *C, uint32_t M, uint32_t N,
                      uint32_t P) {
  start_timer(&t);
  for (uint32_t blockRow = 0; blockRow < cdiv(M, BLOCK_DIM); ++blockRow) {
    for (uint32_t blockCol = 0; blockCol < cdiv(P, BLOCK_DIM); ++blockCol) {
      for (uint32_t row = blockRow * BLOCK_DIM;
           row < (blockRow + 1) * BLOCK_DIM; ++row) {
        for (uint32_t col = blockCol * BLOCK_DIM;
             col < (blockCol + 1) * BLOCK_DIM; ++col) {
          if (row >= M || col >= P)
            continue;

          float sum = 0.0f;
          for (uint32_t block = 0; block < cdiv(N, BLOCK_DIM); ++block) {
            for (uint32_t i = 0; i < BLOCK_DIM; ++i) {
              if (block * BLOCK_DIM + i < N)
                sum += A[row * N + block * BLOCK_DIM + i] *
                       B[(block * BLOCK_DIM + i) * P + col];
            }
          }
          C[row * P + col] = sum;
        }
      }
    }
  }
  stop_timer(&t);
  printf("CPU time(tiled): %f\n", time_diff(&t));
}

bool all_close(float *A, float *B, uint32_t M, uint32_t N) {
  for (uint32_t row = 0; row < M; ++row) {
    for (uint32_t col = 0; col < N; ++col) {
      if (abs(A[row * N + col] - B[row * N + col]) > 1e-3) {
        printf("Mismatch at (%d, %d): A = %f, B = %f\n", row, col,
               A[row * N + col], B[row * N + col]);
        return false;
      }
    }
  }
  return true;
}

void print_mat(float *A, uint32_t M, uint32_t N) {
  printf("[\n");
  for (uint32_t row = 0; row < M; ++row) {
    for (uint32_t col = 0; col < N; ++col) {
      printf("  %f ", A[row * N + col]);
    }
    printf("\n");
  }
  printf("]\n");
}

int main() {
  cudaDeviceSynchronize();

  uint32_t M = 1027, N = 1529, P = 2747;
  // uint32_t M = 127, N = 159, P = 27;
  float *A, *B, *C_cpu, *C_cpu_tiled, *C_gpu, *C_gpu_tiled;

  A = (float *)malloc((M * N) * sizeof(float));
  B = (float *)malloc((N * P) * sizeof(float));
  C_cpu = (float *)malloc((M * P) * sizeof(float));
  C_cpu_tiled = (float *)malloc((M * P) * sizeof(float));
  C_gpu = (float *)malloc((M * P) * sizeof(float));
  C_gpu_tiled = (float *)malloc((M * P) * sizeof(float));

  for (uint32_t row = 0; row < M; ++row) {
    for (uint32_t col = 0; col < N; ++col) {
      A[row * N + col] = (float)rand() / RAND_MAX;
    }
  }
  for (uint32_t row = 0; row < N; ++row) {
    for (uint32_t col = 0; col < P; ++col) {
      B[row * P + col] = (float)rand() / RAND_MAX;
    }
  }

  matmul_cpu(A, B, C_cpu, M, N, P);

  tiled_matmul_cpu(A, B, C_cpu_tiled, M, N, P);

  matmul_gpu(A, B, C_gpu, M, N, P);

  tiled_matmul_gpu(A, B, C_gpu_tiled, M, N, P);

  // print_mat(A, M, N);
  // print_mat(B, N, P);
  // print_mat(C_cpu, M, P);
  // print_mat(C_cpu_tiled, M, P);
  // print_mat(C_gpu, M, P);
  // print_mat(C_gpu_tiled, M, P);

  printf("CPU tiled impl match: %s\n",
         all_close(C_cpu, C_cpu_tiled, M, P) ? "true" : "false");
  printf("GPU impl match: %s\n",
         all_close(C_cpu, C_gpu, M, P) ? "true" : "false");
  printf("GPU tiled impl match: %s\n",
         all_close(C_cpu, C_gpu_tiled, M, P) ? "true" : "false");

  return 0;
}
