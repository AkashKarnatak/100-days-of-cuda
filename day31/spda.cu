#include <stdio.h>

size_t cdiv(size_t a, size_t b) { return (a + b - 1) / b; }

__global__ void spda_kernel(float *__restrict__ output,
                            float *__restrict__ temp, float *__restrict__ query,
                            float *__restrict__ key, float *__restrict__ value,
                            size_t N, size_t M) {
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  size_t tid = threadIdx.x;

  // FIXME: right now i am assuming N == M and N <= 1024

  if (row >= N || col >= M) // FIXME: this condition needs modification
    return;

  float sum = 0;
  for (size_t k = 0; k < M; ++k) {
    sum += query[row * M + k] * key[col * M + k];
  }
  temp[row * N + col] = sum / sqrtf(M);

  __shared__ float global_sum[1];

  if (tid == 0) {
    float sum = 0, maximum = -INFINITY;
    for (size_t k = 0; k < N; ++k) {
      // printf("%lu: %f\n", k, temp[row * N + k]);
      float curr = temp[row * N + k];
      maximum = max(maximum, curr);
    }
    // printf("row: %lu, max %f\n", row, maximum);
    for (size_t k = 0; k < N; ++k) {
      float curr = temp[row * N + k];
      sum += expf(curr - maximum);
    }
    // for (size_t k = 0; k < N; ++k) {
    //   float curr = temp[row * N + k];
    //   if (curr > maximum) {
    //     sum *= expf(maximum - curr);
    //     maximum = curr;
    //   }
    //   sum += expf(curr - maximum);
    // }
    global_sum[0] = sum;
    printf("row: %lu, max %f\n", row, global_sum[0]);
  }

  __syncthreads();

  if (tid == 0)
    printf("row: %lu, max %f\n", row, global_sum[0]);

  // temp[row * N + col] = expf(temp[row * N + col]) / global_sum;

  sum = 0;
  for (size_t k = 0; k < N; ++k) {
    sum += temp[row * N + k] * value[k * M + col];
  }
  output[row * M + col] = sum;
}

void spda_gpu(float *output, float *temp, float *__restrict__ query,
              float *__restrict__ key, float *__restrict__ value, size_t N,
              size_t M) {
  float *query_d, *key_d, *value_d, *temp_d, *output_d;

  cudaMalloc(&query_d, (N * M) * sizeof(float));
  cudaMalloc(&key_d, (N * M) * sizeof(float));
  cudaMalloc(&value_d, (N * M) * sizeof(float));
  cudaMalloc(&temp_d, (N * N) * sizeof(float));
  cudaMalloc(&output_d, (N * M) * sizeof(float));

  cudaMemcpy(query_d, query, (N * M) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(key_d, key, (N * M) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(value_d, value, (N * M) * sizeof(float), cudaMemcpyHostToDevice);

  dim3 numThreads(32, 32);
  dim3 numBlocks(cdiv(M, numThreads.x), cdiv(N, numThreads.y));
  spda_kernel<<<numBlocks, numThreads, N * sizeof(float)>>>(
      output_d, temp_d, query_d, key_d, value_d, N, M);

  cudaMemcpy(output, output_d, (N * M) * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(temp, temp_d, (N * N) * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(query_d);
  cudaFree(key_d);
  cudaFree(value_d);
  cudaFree(temp_d);
  cudaFree(output_d);
}

int main() {
  size_t N, M;
  float *query, *key, *value, *output, *temp;

  N = 8, M = 8;

  query = (float *)malloc((N * M) * sizeof(float));
  key = (float *)malloc((N * M) * sizeof(float));
  value = (float *)malloc((N * M) * sizeof(float));
  output = (float *)malloc((N * M) * sizeof(float));
  temp = (float *)malloc((N * N) * sizeof(float));

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
  //     printf(" %.4f", key[i * M + j]);
  //   }
  //   printf("\n");
  // }
  // printf("\n");

  spda_gpu(output, temp, query, key, value, N, M);

  // for (size_t i = 0; i < N; ++i) {
  //   for (size_t j = 0; j < N; ++j) {
  //     printf(" %.4f", temp[i * N + j]);
  //   }
  //   printf("\n");
  // }
  // printf("\n");

  // for (size_t i = 0; i < N; ++i) {
  //   for (size_t j = 0; j < M; ++j) {
  //     printf(" %.4f", output[i * M + j]);
  //   }
  //   printf("\n");
  // }

  free(query);
  free(key);
  free(value);

  return 0;
}
