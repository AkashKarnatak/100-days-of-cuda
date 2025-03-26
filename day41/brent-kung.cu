#include <stdio.h>
#include <time.h>

#define BLOCK_DIM 1024

inline size_t cdiv(size_t a, size_t b) { return (a + b - 1) / b; }

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

__global__ void brent_kung_inclusive_kernel(float *in, float *out,
                                            float *partial_sums, size_t N) {
  size_t offset = blockIdx.x * blockDim.x * 2;
  size_t tid = threadIdx.x;

  __shared__ float sram[BLOCK_DIM * 2];

  if (offset + tid < N)
    sram[tid] = in[offset + tid];
  else
    sram[tid] = 0.0f;
  if (offset + BLOCK_DIM + tid < N)
    sram[BLOCK_DIM + tid] = in[offset + BLOCK_DIM + tid];
  else
    sram[tid + BLOCK_DIM] = 0.0f;

  __syncthreads();

  // reduction step
  for (size_t s = 1; s <= BLOCK_DIM; s *= 2) {
    size_t j = (tid + 1) * 2 * s - 1;
    if (j < 2 * BLOCK_DIM)
      sram[j] += sram[j - s];
    __syncthreads();
  }

  // post reduction step
  for (size_t s = BLOCK_DIM / 2; s > 0; s /= 2) {
    size_t j = (tid + 1) * 2 * s - 1;
    if (j + s < 2 * BLOCK_DIM)
      sram[j + s] += sram[j];
    __syncthreads();
  }

  if (offset + tid < N)
    out[offset + tid] = sram[tid];
  if (offset + BLOCK_DIM + tid < N)
    out[offset + BLOCK_DIM + tid] = sram[BLOCK_DIM + tid];

  if (tid == BLOCK_DIM - 1) {
    partial_sums[blockIdx.x] = sram[2 * BLOCK_DIM - 1];
  }
}

__global__ void brent_kung_exclusive_kernel(float *in, float *out,
                                            float *partial_sums, size_t N) {
  size_t offset = blockIdx.x * blockDim.x * 2;
  size_t tid = threadIdx.x;

  __shared__ float sram[BLOCK_DIM * 2];

  if (tid > 0 && offset + tid < N)
    sram[tid] = in[offset + tid - 1];
  else
    sram[tid] = 0.0f;
  if (offset + BLOCK_DIM + tid < N)
    sram[BLOCK_DIM + tid] = in[offset + BLOCK_DIM + tid - 1];
  else
    sram[tid + BLOCK_DIM] = 0.0f;

  __syncthreads();

  // reduction step
  for (size_t s = 1; s <= BLOCK_DIM; s *= 2) {
    size_t j = (tid + 1) * 2 * s - 1;
    if (j < 2 * BLOCK_DIM)
      sram[j] += sram[j - s];
    __syncthreads();
  }

  // post reduction step
  for (size_t s = BLOCK_DIM / 2; s > 0; s /= 2) {
    size_t j = (tid + 1) * 2 * s - 1;
    if (j + s < 2 * BLOCK_DIM)
      sram[j + s] += sram[j];
    __syncthreads();
  }

  if (offset + tid < N)
    out[offset + tid] = sram[tid];
  if (offset + BLOCK_DIM + tid < N)
    out[offset + BLOCK_DIM + tid] = sram[BLOCK_DIM + tid];

  if (tid == BLOCK_DIM - 1) {
    float last = N - 1 <= offset + BLOCK_DIM + tid
                     ? in[N - 1]
                     : in[offset + BLOCK_DIM + tid];
    partial_sums[blockIdx.x] = sram[2 * BLOCK_DIM - 1] + last;
  }
}

__global__ void add_inclusive_kernel(float *out, float *partial_sums,
                                     size_t N) {
  size_t offset = blockIdx.x * blockDim.x * 2;
  size_t tid = threadIdx.x;

  if (blockIdx.x == 0)
    return;

  if (offset + tid < N)
    out[offset + tid] += partial_sums[blockIdx.x - 1];
  if (offset + BLOCK_DIM + tid < N)
    out[offset + BLOCK_DIM + tid] += partial_sums[blockIdx.x - 1];
}

__global__ void add_exclusive_kernel(float *out, float *partial_sums,
                                     size_t N) {
  size_t offset = blockIdx.x * blockDim.x * 2;
  size_t tid = threadIdx.x;

  if (blockIdx.x == 0)
    return;

  if (offset + tid < N)
    out[offset + tid] += partial_sums[blockIdx.x];
  if (offset + BLOCK_DIM + tid < N)
    out[offset + BLOCK_DIM + tid] += partial_sums[blockIdx.x];
}

void brent_kung_inclusive_gpu_d(float *in_d, float *out_d, size_t N) {
  float *partial_sums_d;

  size_t numThreads = BLOCK_DIM;
  size_t numBlocks = cdiv(N, numThreads * 2);

  cudaMalloc(&partial_sums_d, numBlocks * sizeof(float));

  brent_kung_inclusive_kernel<<<numBlocks, numThreads>>>(in_d, out_d,
                                                         partial_sums_d, N);
  cudaDeviceSynchronize();

  if (numBlocks > 1)
    brent_kung_inclusive_gpu_d(partial_sums_d, partial_sums_d, numBlocks);

  add_inclusive_kernel<<<numBlocks, numThreads>>>(out_d, partial_sums_d, N);
  cudaDeviceSynchronize();

  cudaFree(partial_sums_d);
}

void brent_kung_exclusive_gpu_d(float *in_d, float *out_d, size_t N) {
  float *partial_sums_d;

  size_t numThreads = BLOCK_DIM;
  size_t numBlocks = cdiv(N, numThreads * 2);

  cudaMalloc(&partial_sums_d, numBlocks * sizeof(float));

  brent_kung_exclusive_kernel<<<numBlocks, numThreads>>>(in_d, out_d,
                                                         partial_sums_d, N);
  cudaDeviceSynchronize();

  if (numBlocks > 1)
    brent_kung_exclusive_gpu_d(partial_sums_d, partial_sums_d, numBlocks);

  add_exclusive_kernel<<<numBlocks, numThreads>>>(out_d, partial_sums_d, N);
  cudaDeviceSynchronize();

  cudaFree(partial_sums_d);
}

void brent_kung_inclusive_gpu(float *in, float *out, size_t N) {
  float *in_d, *out_d;

  cudaMalloc(&in_d, N * sizeof(float));
  cudaMalloc(&out_d, N * sizeof(float));

  cudaMemcpy(in_d, in, N * sizeof(float), cudaMemcpyHostToDevice);

  start_timer(&t);
  brent_kung_inclusive_gpu_d(in_d, out_d, N);
  stop_timer(&t);
  printf("GPU time: %f\n", time_diff(&t));

  cudaMemcpy(out, out_d, N * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(in_d);
  cudaFree(out_d);
}

void brent_kung_exclusive_gpu(float *in, float *out, size_t N) {
  float *in_d, *out_d;

  cudaMalloc(&in_d, N * sizeof(float));
  cudaMalloc(&out_d, N * sizeof(float));

  cudaMemcpy(in_d, in, N * sizeof(float), cudaMemcpyHostToDevice);

  start_timer(&t);
  brent_kung_exclusive_gpu_d(in_d, out_d, N);
  stop_timer(&t);
  printf("GPU time: %f\n", time_diff(&t));

  cudaMemcpy(out, out_d, N * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(in_d);
  cudaFree(out_d);
}

void brent_kung_inclusive_cpu(float *in, float *out, size_t N) {
  start_timer(&t);
  out[0] = in[0];
  for (size_t i = 1; i < N; ++i) {
    out[i] = out[i - 1] + in[i];
  }
  stop_timer(&t);
  printf("CPU time: %f\n", time_diff(&t));
}

void brent_kung_exclusive_cpu(float *in, float *out, size_t N) {
  start_timer(&t);
  out[0] = 0;
  for (size_t i = 1; i < N; ++i) {
    out[i] = out[i - 1] + in[i - 1];
  }
  stop_timer(&t);
  printf("CPU time: %f\n", time_diff(&t));
}

bool allclose(float *A, float *B, size_t N) {
  for (size_t i = 0; i < N; ++i) {
    if (abs(A[i] - B[i]) > 6) {
      printf("Mismatch at (%ld,): A = %f and B = %f\n", i, A[i], B[i]);
      return false;
    }
  }
  return true;
}

void print(float *A, size_t N) {
  printf("[\n");
  for (size_t i = 0; i < N; ++i) {
    printf(" %f", A[i]);
  }
  printf("\n]\n");
}

int main() {
  size_t N;
  float *in, *out_cpu, *out_gpu;

  N = 1'000'000;
  in = (float *)malloc(N * sizeof(float));
  out_cpu = (float *)malloc(N * sizeof(float));
  out_gpu = (float *)malloc(N * sizeof(float));

  for (size_t i = 0; i < N; ++i) {
    in[i] = (float)rand() / RAND_MAX;
    // in[i] = 0.1;
  }

  brent_kung_exclusive_cpu(in, out_cpu, N);

  brent_kung_exclusive_gpu(in, out_gpu, N);

  // print(out_cpu, N);
  // print(out_gpu, N);

  printf("Match impl: %s\n", allclose(out_cpu, out_gpu, N) ? "true" : "false");

  return 0;
}
