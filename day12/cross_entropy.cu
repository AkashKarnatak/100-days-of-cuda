#include <stdio.h>
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

size_t cdiv(size_t a, size_t b) { return (a + b - 1) / b; }

__global__ void cross_entropy_kernel(float *pred_data, size_t *target_data,
                                     float *out_data, size_t N, size_t M) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= N)
    return;

  float sum, maximum;

  maximum = -INFINITY;
  for (size_t k = 0; k < M; ++k) {
    if (pred_data[i * M + k] > maximum)
      maximum = pred_data[i * M + k];
  }

  sum = 0.0f;
  for (size_t k = 0; k < M; ++k) {
    sum += expf(pred_data[i * M + k] - maximum);
  }

  out_data[i] = -log(expf(pred_data[i * M + target_data[i]] - maximum) / sum);
}

void cross_entropy_gpu(float *pred_data, size_t *target_data, float *out_data,
                       size_t N, size_t M) {
  float *pred_data_d, *out_data_d;
  size_t *target_data_d;

  // allocate memory on GPU
  start_timer(&t);
  cudaMalloc(&pred_data_d, N * M * sizeof(float));
  cudaMalloc(&target_data_d, N * sizeof(size_t));
  cudaMalloc(&out_data_d, N * sizeof(float));

  // copy data from host to device
  cudaMemcpy(pred_data_d, pred_data, N * M * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(target_data_d, target_data, N * sizeof(size_t),
             cudaMemcpyHostToDevice);
  stop_timer(&t);
  printf("CPU to GPU copy time: %f\n", time_diff(&t));

  // perform computation
  size_t numThreads = 1024;
  size_t numBlocks = cdiv(N, numThreads);
  start_timer(&t);
  cudaDeviceSynchronize();
  cross_entropy_kernel<<<numBlocks, numThreads>>>(pred_data_d, target_data_d,
                                                  out_data_d, N, M);
  cudaDeviceSynchronize();
  stop_timer(&t);
  printf("GPU time: %f\n", time_diff(&t));

  // copy data from device to host
  cudaMemcpy(out_data, out_data_d, N * sizeof(float), cudaMemcpyDeviceToHost);

  // free memory
  cudaFree(pred_data_d);
  cudaFree(target_data_d);
  cudaFree(out_data_d);
}

void cross_entropy_cpu(float *pred_data, size_t *target_data, float *out_data,
                       size_t N, size_t M) {
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

    out_data[row] =
        -log(expf(pred_data[row * M + target_data[row]] - maximum) / sum);
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

  float *pred_data, *out_data_cpu, *out_data_gpu;
  size_t *target_data;
  size_t N, M;

  N = 10240, M = 10240;

  pred_data = (float *)malloc(N * M * sizeof(float));
  target_data = (size_t *)malloc(N * sizeof(size_t));
  out_data_cpu = (float *)malloc(N * sizeof(float));
  out_data_gpu = (float *)malloc(N * sizeof(float));

  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < M; ++j) {
      pred_data[i * M + j] = (float)rand() / RAND_MAX;
    }
  }
  for (size_t i = 0; i < N; ++i) {
    target_data[i] = rand() % M;
  }

  cross_entropy_cpu(pred_data, target_data, out_data_cpu, N, M);

  cross_entropy_gpu(pred_data, target_data, out_data_gpu, N, M);

  printf("CPU and GPU match: %s\n",
         allclose(out_data_cpu, out_data_gpu, N) ? "true" : "false");

  return 0;
}
