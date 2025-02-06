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

int32_t main() {
  int32_t N = 100000000;

  cudaDeviceSynchronize();

  int32_t *x = (int32_t *)malloc(N * sizeof(int32_t));
  for (int32_t i = 0; i < N; ++i) {
    x[i] = rand();
  }

  start_timer(&t);
  int32_t *x_h;
  x_h = (int32_t *)malloc(N * sizeof(int32_t));
  memcpy(x_h, x, N * sizeof(int32_t));
  stop_timer(&t);
  printf("CPU to CPU copy time: %f\n", time_diff(&t));

  start_timer(&t);
  int32_t *x_d;
  cudaMalloc(&x_d, N * sizeof(int32_t));
  cudaMemcpy(x_d, x, N * sizeof(int32_t), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  stop_timer(&t);
  printf("CPU to GPU copy time: %f\n", time_diff(&t));

  // whatever i put second takes more time, idk why
  // so ig, cudaMemcpy = cudaHostRegister + cudaMemcpy + cudaHostUnregister
  start_timer(&t);
  int32_t *x_p;
  cudaMalloc(&x_p, N * sizeof(int32_t));
  cudaHostRegister(&x, N * sizeof(int32_t), cudaHostRegisterDefault);
  cudaMemcpy(x_p, x, N * sizeof(int32_t), cudaMemcpyHostToDevice);
  cudaHostUnregister(x_p);
  cudaDeviceSynchronize();
  stop_timer(&t);
  printf("CPU to GPU copy time(manual pinning): %f\n", time_diff(&t));
}
