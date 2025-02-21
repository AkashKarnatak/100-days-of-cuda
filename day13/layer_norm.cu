#include <stdio.h>
#include <time.h>

#define BLOCK_SIZE 1024

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

__global__ void naive_layer_norm_kernel(float *in_data, float *out_data,
                                        size_t N, size_t M) {
  size_t row = blockDim.x * blockIdx.x + threadIdx.x;
  if (row >= N)
    return;

  float sum, sum2, mean, std;

  sum = 0.0f, sum2 = 0.0f;
  for (size_t col = 0; col < M; ++col) {
    float curr = in_data[row * M + col];
    sum += curr;
    sum2 += curr * curr;
  }

  mean = sum / M;
  std = sqrt(sum2 / M - mean * mean + 1e-5);

  for (size_t col = 0; col < M; ++col) {
    out_data[row * M + col] = (in_data[row * M + col] - mean) / std;
  }
}

__global__ void block_reduction_layer_norm_kernel(float *in_data,
                                                  float *out_data, size_t N,
                                                  size_t M) {
  size_t row = blockIdx.x;

  float partial_sum, partial_sum2, mean, std;

  __shared__ float sums_s[BLOCK_SIZE];
  __shared__ float sum2s_s[BLOCK_SIZE];

  partial_sum = 0.0f, partial_sum2 = 0.0f;
  for (size_t block = 0; block < cdiv(M, BLOCK_SIZE); ++block) {
    if (block * BLOCK_SIZE + threadIdx.x < M) {
      float curr = in_data[row * M + block * BLOCK_SIZE + threadIdx.x];
      partial_sum += curr;
      partial_sum2 += curr * curr;
    }
  }

  sums_s[threadIdx.x] = partial_sum;
  sum2s_s[threadIdx.x] = partial_sum2;

  __syncthreads();

  for (size_t numThreads = blockDim.x / 2; numThreads > 0; numThreads /= 2) {
    if (threadIdx.x < numThreads) {
      sums_s[threadIdx.x] += sums_s[threadIdx.x + numThreads];
      sum2s_s[threadIdx.x] += sum2s_s[threadIdx.x + numThreads];
    }
    __syncthreads();
  }

  mean = sums_s[0] / M;
  std = sqrt(sum2s_s[0] / M - mean * mean + 1e-5f);

  for (size_t block = 0; block < cdiv(M, BLOCK_SIZE); ++block) {
    if (block * BLOCK_SIZE + threadIdx.x < M) {
      out_data[row * M + block * BLOCK_SIZE + threadIdx.x] =
          (in_data[row * M + block * BLOCK_SIZE + threadIdx.x] - mean) / std;
    }
  }
}

void naive_layer_norm_gpu(float *in_data, float *out_data, size_t N, size_t M) {
  float *in_data_d, *out_data_d;

  // allocate memory on GPU
  cudaMalloc(&in_data_d, N * M * sizeof(float));
  cudaMalloc(&out_data_d, N * M * sizeof(float));

  // copy data from host to device
  cudaMemcpy(in_data_d, in_data, N * M * sizeof(float), cudaMemcpyHostToDevice);

  // perform computation
  size_t numThreads = 1024;
  size_t numBlocks = cdiv(N, numThreads);
  start_timer(&t);
  cudaDeviceSynchronize();
  naive_layer_norm_kernel<<<numBlocks, numThreads>>>(in_data_d, out_data_d, N,
                                                     M);
  cudaDeviceSynchronize();
  stop_timer(&t);
  printf("GPU time (Naive): %f\n", time_diff(&t));

  // copy data from device to host
  cudaMemcpy(out_data, out_data_d, N * M * sizeof(float),
             cudaMemcpyDeviceToHost);

  // free memory
  cudaFree(in_data_d);
  cudaFree(out_data_d);
}

void block_reduction_layer_norm_gpu(float *in_data, float *out_data, size_t N,
                                    size_t M) {
  float *in_data_d, *out_data_d;

  // allocate memory on GPU
  cudaMalloc(&in_data_d, N * M * sizeof(float));
  cudaMalloc(&out_data_d, N * M * sizeof(float));

  // copy data from host to device
  cudaMemcpy(in_data_d, in_data, N * M * sizeof(float), cudaMemcpyHostToDevice);

  // perform computation
  size_t numThreads = 1024;
  size_t numBlocks = N;
  start_timer(&t);
  cudaDeviceSynchronize();
  block_reduction_layer_norm_kernel<<<numBlocks, numThreads>>>(
      in_data_d, out_data_d, N, M);
  cudaDeviceSynchronize();
  stop_timer(&t);
  printf("GPU time (Block reduction): %f\n", time_diff(&t));

  // copy data from device to host
  cudaMemcpy(out_data, out_data_d, N * M * sizeof(float),
             cudaMemcpyDeviceToHost);

  // free memory
  cudaFree(in_data_d);
  cudaFree(out_data_d);
}

void layer_norm_cpu(float *in_data, float *out_data, size_t N, size_t M) {
  start_timer(&t);
  for (size_t row = 0; row < N; ++row) {
    for (size_t row = 0; row < N; ++row) {
      float sum, sum2, mean, std;

      sum = 0.0f, sum2 = 0.0f;
      for (size_t col = 0; col < M; ++col) {
        sum += in_data[row * M + col];
        sum2 += in_data[row * M + col] * in_data[row * M + col];
      }

      mean = sum / M;
      std = sqrt(sum2 / M - mean * mean + 1e-5);
      for (size_t col = 0; col < M; ++col) {
        out_data[row * M + col] = (in_data[row * M + col] - mean) / std;
      }
    }
  }
  stop_timer(&t);
  printf("CPU time: %f\n", time_diff(&t));
}

bool allclose(float *a, float *b, size_t N) {
  for (size_t i = 0; i < N; ++i) {
    if (abs(a[i] - b[i]) > 1e-4)
      return false;
  }
  return true;
}

int32_t main() {
  cudaDeviceSynchronize();

  float *in_data, *out_data_cpu, *out_data_gpu;
  size_t N, M;

  N = 2048, M = 2048;

  in_data = (float *)malloc(N * M * sizeof(float));
  out_data_cpu = (float *)malloc(N * M * sizeof(float));
  out_data_gpu = (float *)malloc(N * M * sizeof(float));

  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < M; ++j) {
      in_data[i * M + j] = (float)rand() / RAND_MAX;
    }
  }

  layer_norm_cpu(in_data, out_data_cpu, N, M);

  naive_layer_norm_gpu(in_data, out_data_gpu, N, M);

  printf("CPU and GPU match(Naive): %s\n",
         allclose(out_data_cpu, out_data_gpu, N * M) ? "true" : "false");

  block_reduction_layer_norm_gpu(in_data, out_data_gpu, N, M);

  printf("CPU and GPU match(Block reduction): %s\n",
         allclose(out_data_cpu, out_data_gpu, N * M) ? "true" : "false");

  return 0;
}
