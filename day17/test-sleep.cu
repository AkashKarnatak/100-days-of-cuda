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

__global__ void kernel() {
  for (int i = 0; i < 1000; i++)
    __nanosleep(1000000U);
}

int main() {
  cudaDeviceSynchronize();

  float *a, *b;
  float av = 4, bv = 5;
  cudaDeviceProp devProp;

  cudaMalloc(&a, sizeof(float));
  cudaMalloc(&b, sizeof(float));
  cudaMemcpy(a, &av, sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(b, &bv, sizeof(float), cudaMemcpyHostToDevice);

  cudaGetDeviceProperties(&devProp, 0);

  printf("Number of SMs: %d\n", devProp.multiProcessorCount);
  printf("Max threads per block: %d\n", devProp.maxThreadsPerBlock);
  printf("Max blocks per SM: %d\n", devProp.maxBlocksPerMultiProcessor);
  printf("Max threads per SM: %d\n\n", devProp.maxThreadsPerMultiProcessor);

  start_timer(&t);
  kernel<<<31, 1024>>>();
  cudaDeviceSynchronize();
  stop_timer(&t);
  printf("Total threads: %d, Time took: %f\n", 31 * 1024, time_diff(&t));

  start_timer(&t);
  kernel<<<42, 768>>>();
  cudaDeviceSynchronize();
  stop_timer(&t);
  printf("Total threads: %d, Time took: %f\n", 42 * 768, time_diff(&t));

  return 0;
}
