#include <stdio.h>
#include <time.h>

#define C0 1
#define C1 1

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

__host__ __device__ size_t cdiv(size_t a, size_t b) { return (a + b - 1) / b; }

__global__ void stencil3d_kernel(float *in_data, float *out_data, size_t C,
                                 size_t N, size_t M) {
  size_t channel = blockDim.z * blockIdx.z + threadIdx.z;
  size_t row = blockDim.y * blockIdx.y + threadIdx.y;
  size_t col = blockDim.x * blockIdx.x + threadIdx.x;
  if (channel < 1 || channel >= C - 1 || row < 1 || row >= N - 1 || col < 1 ||
      col >= M - 1)
    return;

  out_data[channel * N * M + row * M + col] =
      C0 * in_data[channel * N * M + row * M + col] +
      C1 * (in_data[(channel - 1) * N * M + row * M + col] +
            in_data[(channel + 1) * N * M + row * M + col] +
            in_data[channel * N * M + (row - 1) * M + col] +
            in_data[channel * N * M + (row + 1) * M + col] +
            in_data[channel * N * M + row * M + (col - 1)] +
            in_data[channel * N * M + row * M + (col + 1)]);
}

void stencil3d_gpu(float *in_data, float *out_data, size_t C, size_t N,
                   size_t M) {
  float *in_data_d, *out_data_d;

  cudaMalloc(&in_data_d, C * N * M * sizeof(float));
  cudaMalloc(&out_data_d, C * N * M * sizeof(float));

  cudaMemcpy(in_data_d, in_data, C * N * M * sizeof(float),
             cudaMemcpyHostToDevice);

  dim3 numThreads(8, 8, 8);
  dim3 numBlocks(cdiv(M, numThreads.x), cdiv(N, numThreads.y),
                 cdiv(C, numThreads.z));
  start_timer(&t);
  stencil3d_kernel<<<numBlocks, numThreads>>>(in_data_d, out_data_d, C, N, M);
  cudaDeviceSynchronize();
  stop_timer(&t);
  printf("GPU time: %f\n", time_diff(&t));

  cudaMemcpy(out_data, out_data_d, C * N * M * sizeof(float),
             cudaMemcpyDeviceToHost);

  cudaFree(in_data_d);
  cudaFree(out_data_d);
}

void stencil3d_cpu(float *in_data, float *out_data, size_t C, size_t N,
                   size_t M) {
  start_timer(&t);
  for (size_t c = 0; c < C; ++c) {
    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < M; ++j) {
        if (c >= 1 && c < C - 1 && i >= 1 && i < N - 1 && j >= 1 && j < M - 1) {
          out_data[c * N * M + i * M + j] =
              C0 * in_data[c * N * M + i * M + j] +
              C1 * (in_data[(c - 1) * N * M + i * M + j] +
                    in_data[(c + 1) * N * M + i * M + j] +
                    in_data[c * N * M + (i - 1) * M + j] +
                    in_data[c * N * M + (i + 1) * M + j] +
                    in_data[c * N * M + i * M + (j - 1)] +
                    in_data[c * N * M + i * M + (j + 1)]);
        }
      }
    }
  }
  stop_timer(&t);
  printf("CPU time: %f\n", time_diff(&t));
}

bool allclose(float *A, float *B, size_t C, size_t N, size_t M) {
  for (size_t c = 0; c < C; ++c) {
    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < M; ++j) {
        if (abs(A[c * N * M + i * M + j] - B[c * N * M + i * M + j]) > 1e-4) {
          printf("Mismatch at (%lu, %lu, %lu), A = %f and B = %f\n", c, i, j,
                 A[c * N * M + i * M + j], B[c * N * M + i * M + j]);
          return false;
        }
      }
    }
  }
  return true;
}

int main() {
  cudaDeviceSynchronize();

  float *in_data, *out_data_cpu, *out_data_gpu;
  size_t C, N, M;

  C = 128, N = 128, M = 128;
  in_data = (float *)malloc(C * N * M * sizeof(float));
  out_data_cpu = (float *)malloc(C * N * M * sizeof(float));
  out_data_gpu = (float *)malloc(C * N * M * sizeof(float));

  for (size_t c = 0; c < C; ++c) {
    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < M; ++j) {
        in_data[c * N * M + i * M + j] = (float)rand() / RAND_MAX;
      }
    }
  }
  memset(out_data_cpu, 0, C * M * N * sizeof(float));
  memset(out_data_gpu, 0, C * M * N * sizeof(float));

  stencil3d_cpu(in_data, out_data_cpu, C, N, M);

  stencil3d_gpu(in_data, out_data_gpu, C, N, M);

  printf("GPU and CPU impl match: %s\n",
         allclose(out_data_cpu, out_data_gpu, C, N, M) ? "true" : "false");

  free(in_data);
  free(out_data_cpu);
  free(out_data_gpu);

  return 0;
}
