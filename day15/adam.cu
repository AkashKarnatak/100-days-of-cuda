#include <cstring>
#include <omp.h>
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

struct timer timer;

size_t cdiv(size_t a, size_t b) { return (a + b - 1) / b; }

__global__ void adam_update_kernel(float *weights, float *grads, float *m,
                                   float *v, size_t N, float beta1, float beta2,
                                   float lr, float eps, size_t t) {
  float m_hat, v_hat;

  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= N)
    return;

  m[i] = beta1 * m[i] + (1.0f - beta1) * grads[i];
  v[i] = beta2 * v[i] + (1.0f - beta2) * grads[i] * grads[i];
  m_hat = m[i] / (1.0f - powf(beta1, t));
  v_hat = v[i] / (1.0f - powf(beta2, t));
  weights[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
}

void adam_update_gpu(float *weights, float *grads, float *m, float *v, size_t N,
                     float beta1, float beta2, float lr, float eps, size_t t) {
  float *weights_d, *grads_d, *m_d, *v_d;

  // allocate memory on GPU
  cudaMalloc(&weights_d, N * sizeof(float));
  cudaMalloc(&grads_d, N * sizeof(float));
  cudaMalloc(&m_d, N * sizeof(float));
  cudaMalloc(&v_d, N * sizeof(float));

  // copy data from host to device
  cudaMemcpy(weights_d, weights, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(grads_d, grads, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(m_d, m, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(v_d, v, N * sizeof(float), cudaMemcpyHostToDevice);

  // perform computation
  size_t numThreads = 1024;
  size_t numBlocks = cdiv(N, numThreads);
  cudaDeviceSynchronize();
  adam_update_kernel<<<numBlocks, numThreads>>>(weights_d, grads_d, m_d, v_d, N,
                                                beta1, beta2, lr, eps, t);
  cudaDeviceSynchronize();

  // copy data from device to host
  cudaMemcpy(weights, weights_d, N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(grads, grads_d, N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(m, m_d, N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(v, v_d, N * sizeof(float), cudaMemcpyDeviceToHost);

  // free memory
  cudaFree(weights_d);
  cudaFree(grads_d);
  cudaFree(m_d);
  cudaFree(v_d);
}

void adam_update_cpu(float *weights, float *grads, float *m, float *v, size_t N,
                     float beta1, float beta2, float lr, float eps, size_t t) {
  float m_hat, v_hat;
#pragma omp parallel for
  for (size_t i = 0; i < N; ++i) {
    m[i] = beta1 * m[i] + (1.0f - beta1) * grads[i];
    v[i] = beta2 * v[i] + (1.0f - beta2) * grads[i] * grads[i];
    m_hat = m[i] / (1.0f - powf(beta1, t));
    v_hat = v[i] / (1.0f - powf(beta2, t));
    weights[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
  }
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

  size_t N;
  float *weights_cpu, *weights_gpu, *grads, *m, *v;
  float lr, beta1, beta2, eps;

  N = 10000000;
  lr = 0.001f, beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;

  weights_cpu = (float *)malloc(N * sizeof(float));
  weights_gpu = (float *)malloc(N * sizeof(float));
  grads = (float *)malloc(N * sizeof(float));
  m = (float *)malloc(N * sizeof(float));
  v = (float *)malloc(N * sizeof(float));

  for (size_t i = 0; i < N; ++i) {
    weights_cpu[i] = weights_gpu[i] = (float)rand() / RAND_MAX;
    grads[i] = (float)rand() / RAND_MAX;
  }
  memset(m, 0, N * sizeof(float));
  memset(v, 0, N * sizeof(float));

  start_timer(&timer);
  for (size_t t = 1; t < 100; ++t) {
    adam_update_cpu(weights_cpu, grads, m, v, N, beta1, beta2, lr, eps, t);
  }
  stop_timer(&timer);
  printf("CPU time: %f\n", time_diff(&timer));

  memset(m, 0, N * sizeof(float));
  memset(v, 0, N * sizeof(float));

  start_timer(&timer);
  for (size_t t = 1; t < 100; ++t) {
    adam_update_gpu(weights_gpu, grads, m, v, N, beta1, beta2, lr, eps, t);
  }
  stop_timer(&timer);
  printf("GPU time: %f\n", time_diff(&timer));

  printf("CPU and GPU match: %s\n",
         allclose(weights_cpu, weights_gpu, N) ? "true" : "false");

  return 0;
}
