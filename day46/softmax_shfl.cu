#include <stdio.h>
#include <time.h>

#define TILE_SIZE 1024

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

__global__ void softmax_kernel(float *in_data, float *out_data, size_t N,
                               size_t M) {
  size_t row = blockIdx.x;

  float local_sum, local_max, global_sum, global_max;
  __shared__ float mem_s[TILE_SIZE];

  local_sum = 0.0f, local_max = -INFINITY;
  for (size_t tileIdx = 0; tileIdx < cdiv(M, TILE_SIZE); ++tileIdx) {
    if (tileIdx * TILE_SIZE + threadIdx.x >= M)
      continue;
    float curr = in_data[row * M + tileIdx * TILE_SIZE + threadIdx.x];
    if (curr > local_max) {
      local_sum *= expf(local_max - curr);
      local_max = curr;
    }
    local_sum += expf(curr - local_max);
  }

  mem_s[threadIdx.x] = local_max;

  __syncthreads();

  for (size_t numThreads = TILE_SIZE / 2; numThreads > 0; numThreads /= 2) {
    if (threadIdx.x < numThreads) {
      mem_s[threadIdx.x] =
          max(mem_s[threadIdx.x], mem_s[threadIdx.x + numThreads]);
    }
    __syncthreads();
  }

  global_max = mem_s[0];

  mem_s[threadIdx.x] = local_sum * expf(local_max - global_max);

  __syncthreads();

  for (size_t numThreads = TILE_SIZE / 2; numThreads > 0; numThreads /= 2) {
    if (threadIdx.x < numThreads) {
      mem_s[threadIdx.x] += mem_s[threadIdx.x + numThreads];
    }
    __syncthreads();
  }

  global_sum = mem_s[0];

  __syncthreads();

  for (size_t tileIdx = 0; tileIdx < cdiv(M, TILE_SIZE); ++tileIdx) {
    if (tileIdx * TILE_SIZE + threadIdx.x < M) {
      out_data[row * M + tileIdx * TILE_SIZE + threadIdx.x] =
          expf(in_data[row * M + tileIdx * TILE_SIZE + threadIdx.x] -
               global_max) /
          global_sum;
    }
  }
}

__global__ void softmax_shfl_kernel(float *in_data, float *out_data, size_t N,
                                    size_t M) {
  size_t row = blockIdx.x;

  float local_sum, local_max, global_sum, global_max;
  __shared__ float mem_s[TILE_SIZE];

  local_sum = 0.0f, local_max = -INFINITY;
  for (size_t tileIdx = 0; tileIdx < cdiv(M, TILE_SIZE); ++tileIdx) {
    if (tileIdx * TILE_SIZE + threadIdx.x >= M)
      continue;
    float curr = in_data[row * M + tileIdx * TILE_SIZE + threadIdx.x];
    if (curr > local_max) {
      local_sum *= expf(local_max - curr);
      local_max = curr;
    }
    local_sum += expf(curr - local_max);
  }

  mem_s[threadIdx.x] = local_max;

  __syncthreads();

  for (size_t numThreads = TILE_SIZE / 2; numThreads > warpSize;
       numThreads /= 2) {
    if (threadIdx.x < numThreads) {
      mem_s[threadIdx.x] =
          max(mem_s[threadIdx.x], mem_s[threadIdx.x + numThreads]);
    }
    __syncthreads();
  }
  float maximum = mem_s[threadIdx.x];
  if (threadIdx.x + warpSize < TILE_SIZE) {
    maximum = max(maximum, mem_s[threadIdx.x + warpSize]);
  }
  for (size_t s = warpSize / 2; s > 0; s /= 2) {
    maximum = max(maximum, __shfl_down_sync(0xffffffff, maximum, s));
  }
  if (threadIdx.x == 0)
    mem_s[0] = maximum;
  __syncthreads();
  global_max = mem_s[0];

  mem_s[threadIdx.x] = local_sum * expf(local_max - global_max);

  __syncthreads();

  for (size_t numThreads = TILE_SIZE / 2; numThreads > warpSize;
       numThreads /= 2) {
    if (threadIdx.x < numThreads) {
      mem_s[threadIdx.x] += mem_s[threadIdx.x + numThreads];
    }
    __syncthreads();
  }
  float sum = mem_s[threadIdx.x];
  if (threadIdx.x + warpSize < TILE_SIZE) {
    sum += mem_s[threadIdx.x + warpSize];
  }
  for (size_t s = warpSize / 2; s > 0; s /= 2) {
    sum += __shfl_down_sync(0xffffffff, sum, s);
  }
  if (threadIdx.x == 0)
    mem_s[0] = sum;
  __syncthreads();
  global_sum = mem_s[0];

  __syncthreads();

  for (size_t tileIdx = 0; tileIdx < cdiv(M, TILE_SIZE); ++tileIdx) {
    if (tileIdx * TILE_SIZE + threadIdx.x < M) {
      out_data[row * M + tileIdx * TILE_SIZE + threadIdx.x] =
          expf(in_data[row * M + tileIdx * TILE_SIZE + threadIdx.x] -
               global_max) /
          global_sum;
    }
  }
}

void softmax_gpu(float *in_data, float *out_data, size_t N, size_t M) {
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
  softmax_kernel<<<numBlocks, numThreads>>>(in_data_d, out_data_d, N, M);
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

void softmax_shfl_gpu(float *in_data, float *out_data, size_t N, size_t M) {
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
  softmax_shfl_kernel<<<numBlocks, numThreads>>>(in_data_d, out_data_d, N, M);
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

void softmax_cpu(float *pred_data, float *out_data, size_t N, size_t M) {
  start_timer(&t);
  for (size_t row = 0; row < N; ++row) {
    float sum, maximum;

    maximum = -INFINITY;
    for (size_t col = 0; col < M; ++col) {
      if (pred_data[row * M + col] > maximum) {
        maximum = pred_data[row * M + col];
      }
    }

    sum = 0.0f;
    for (size_t col = 0; col < M; ++col) {
      sum += expf(pred_data[row * M + col] - maximum);
    }

    for (size_t col = 0; col < M; ++col) {
      out_data[row * M + col] = expf(pred_data[row * M + col] - maximum) / sum;
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

  N = 32786, M = 1024;

  in_data = (float *)malloc(N * M * sizeof(float));
  out_data_cpu = (float *)malloc(N * M * sizeof(float));
  out_data_gpu = (float *)malloc(N * M * sizeof(float));

  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < M; ++j) {
      in_data[i * M + j] = (float)rand() / RAND_MAX;
    }
  }

  softmax_cpu(in_data, out_data_cpu, N, M);

  softmax_gpu(in_data, out_data_gpu, N, M);

  printf("CPU and GPU match: %s\n",
         allclose(out_data_cpu, out_data_gpu, N * M) ? "true" : "false");

  softmax_shfl_gpu(in_data, out_data_gpu, N, M);

  printf("CPU and GPU match (Shfl): %s\n",
         allclose(out_data_cpu, out_data_gpu, N * M) ? "true" : "false");

  return 0;
}
