#include <stdio.h>

size_t cdiv(size_t a, size_t b) { return (a + b - 1) / b; }

__global__ void spda_kernel(float *__restrict__ output,
                            float *__restrict__ weight, float *__restrict__ query,
                            float *__restrict__ key, float *__restrict__ value,
                            size_t N, size_t M) {
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;

  // FIXME: right now i am assuming N == M and N <= 1024

  if (row >= N || col >= M) // FIXME: this condition needs modification
    return;

  float sum = 0;
  for (size_t k = 0; k < M; ++k) {
    sum += query[row * M + k] * key[col * M + k];
  }
  weight[row * N + col] = sum / sqrtf(M);

  __syncthreads();

  extern __shared__ float mem_s[];

  float local_sum = 0, local_max = -INFINITY;
  float global_sum, global_max;

  for (size_t i = threadIdx.x; i < N; i += blockDim.x) {
    float curr = weight[row * N + i];
    if (curr > local_max) {
      local_sum *= expf(local_max - curr);
      local_max = curr;
    }
    local_sum += expf(curr - local_max);
  }
  mem_s[threadIdx.y * blockDim.x + threadIdx.x] = local_max;

  __syncthreads();

  for (size_t s = blockDim.x / 2; s > 0; s /= 2) {
    if (threadIdx.x < s) {
      mem_s[threadIdx.y * blockDim.x + threadIdx.x] =
          max(mem_s[threadIdx.y * blockDim.x + threadIdx.x],
              mem_s[threadIdx.y * blockDim.x + threadIdx.x + s]);
    }
    __syncthreads();
  }

  global_max = mem_s[threadIdx.y * blockDim.x];
  local_sum *= expf(local_max - global_max);
  mem_s[threadIdx.y * blockDim.x + threadIdx.x] = local_sum;

  __syncthreads();

  for (size_t s = blockDim.x / 2; s > 0; s /= 2) {
    if (threadIdx.x < s) {
      mem_s[threadIdx.y * blockDim.x + threadIdx.x] +=
          mem_s[threadIdx.y * blockDim.x + threadIdx.x + s];
    }
    __syncthreads();
  }

  global_sum = mem_s[threadIdx.y * blockDim.x];

  weight[row * N + col] = expf(weight[row * N + col] - global_max) / global_sum;

  __syncthreads();

  sum = 0;
  for (size_t k = 0; k < N; ++k) {
    sum += weight[row * N + k] * value[k * M + col];
  }
  output[row * M + col] = sum;
}

extern "C" {
void spda_gpu(float *output, float *weight, float *__restrict__ query,
              float *__restrict__ key, float *__restrict__ value, size_t N,
              size_t M) {
  float *query_d, *key_d, *value_d, *weight_d, *output_d;

  cudaMalloc(&query_d, (N * M) * sizeof(float));
  cudaMalloc(&key_d, (N * M) * sizeof(float));
  cudaMalloc(&value_d, (N * M) * sizeof(float));
  cudaMalloc(&weight_d, (N * N) * sizeof(float));
  cudaMalloc(&output_d, (N * M) * sizeof(float));

  cudaMemcpy(query_d, query, (N * M) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(key_d, key, (N * M) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(value_d, value, (N * M) * sizeof(float), cudaMemcpyHostToDevice);

  dim3 numThreads(32, 32);
  dim3 numBlocks(cdiv(M, numThreads.x), cdiv(N, numThreads.y));
  spda_kernel<<<numBlocks, numThreads,
                (numThreads.y * numThreads.x) * sizeof(float)>>>(
      output_d, weight_d, query_d, key_d, value_d, N, M);

  cudaMemcpy(output, output_d, (N * M) * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(weight, weight_d, (N * N) * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(query_d);
  cudaFree(key_d);
  cudaFree(value_d);
  cudaFree(weight_d);
  cudaFree(output_d);
}
}

int main() {
  size_t N, M;
  float *query, *key, *value, *output, *weight;

  N = 64, M = 64;

  query = (float *)malloc((N * M) * sizeof(float));
  key = (float *)malloc((N * M) * sizeof(float));
  value = (float *)malloc((N * M) * sizeof(float));
  output = (float *)malloc((N * M) * sizeof(float));
  weight = (float *)malloc((N * N) * sizeof(float));

  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < M; ++j) {
      query[i * M + j] = (float)rand() / RAND_MAX;
      key[i * M + j] = (float)rand() / RAND_MAX;
      value[i * M + j] = (float)rand() / RAND_MAX;
    }
  }

  // for (size_t i = 0; i < N; ++i) {
  //   for (size_t j = 0; j < M; ++j) {
  //     printf(" %.4f", query[i * M + j]);
  //   }
  //   printf("\n");
  // }
  // printf("\n");

  // for (size_t i = 0; i < N; ++i) {
  //   for (size_t j = 0; j < M; ++j) {
  //     printf(" %.4f", value[i * M + j]);
  //   }
  //   printf("\n");
  // }
  // printf("\n");

  spda_gpu(output, weight, query, key, value, N, M);

  // for (size_t i = 0; i < N; ++i) {
  //   for (size_t j = 0; j < N; ++j) {
  //     printf(" %.4f", weight[i * N + j]);
  //   }
  //   printf("\n");
  // }
  // printf("\n");

  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < M; ++j) {
      printf(" %.4f", output[i * M + j]);
    }
    printf("\n");
  }

  free(query);
  free(key);
  free(value);

  return 0;
}
