#include <stdio.h>

#define BLOCK_DIM 1024

inline size_t cdiv(size_t a, size_t b) { return (a + b - 1) / b; }

__global__ void kogge_stone_exclusive_kernel(int32_t *in, int32_t *out,
                                             int32_t *partial_sums, size_t N,
                                             int32_t pos) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t tid = threadIdx.x;

  __shared__ int32_t sram[BLOCK_DIM];

  if (tid > 0 && i < N)
    sram[tid] = pos == -1 ? in[i - 1] : (in[i - 1] & (1 << pos)) > 0;
  else
    sram[tid] = 0.0f;

  __syncthreads();

  for (size_t s = 1; s < BLOCK_DIM; s *= 2) {
    int32_t prev;
    if (tid >= s) {
      prev = sram[tid - s];
    }
    __syncthreads();
    if (tid >= s) {
      sram[tid] = __fadd_rz(sram[tid], prev);
    }
    __syncthreads();
  }

  if (i < N)
    out[i] = sram[tid];

  if (tid == BLOCK_DIM - 1) {
    partial_sums[blockIdx.x] =
        sram[tid] + (pos == -1 ? in[i] : ((in[i] & (1 << pos)) > 0));
  }
}

__global__ void add_exclusive_kernel(int32_t *out, int32_t *partial_sums,
                                     size_t N) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= N)
    return;

  if (blockIdx.x > 0) {
    out[i] += partial_sums[blockIdx.x];
  }
}

__global__ void radix_kernel(int32_t *in, int32_t *left, int32_t *out, size_t N,
                             int32_t pos) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx >= N)
    return;

  size_t dest;
  int32_t x = in[idx];

  if (x & (1 << pos)) {
    dest = N - (left[N - 1] + ((in[N - 1] & (1 << pos)) > 0)) + left[idx];
  } else {
    dest = idx - left[idx];
  }

  out[dest] = x;
}

void kogge_stone_exclusive_gpu_d(int32_t *in_d, int32_t *out_d, size_t N,
                                 int32_t pos) {
  int32_t *partial_sums_d;

  size_t numThreads = BLOCK_DIM;
  size_t numBlocks = cdiv(N, numThreads);

  cudaMallocManaged(&partial_sums_d, numBlocks * sizeof(int32_t));

  kogge_stone_exclusive_kernel<<<numBlocks, numThreads>>>(
      in_d, out_d, partial_sums_d, N, pos);
  cudaDeviceSynchronize();

  if (numBlocks > 1) {
    kogge_stone_exclusive_gpu_d(partial_sums_d, partial_sums_d, numBlocks, -1);
  }

  add_exclusive_kernel<<<numBlocks, numThreads>>>(out_d, partial_sums_d, N);
  cudaDeviceSynchronize();

  cudaFree(partial_sums_d);
}

void kogge_stone_exclusive_gpu(int32_t *in, int32_t *out, size_t N,
                               int32_t pos) {
  int32_t *in_d, *out_d;

  size_t numThreads = BLOCK_DIM;
  size_t numBlocks = cdiv(N, numThreads);

  cudaMalloc(&in_d, N * sizeof(int32_t));
  cudaMalloc(&out_d, N * sizeof(int32_t));

  cudaMemcpy(in_d, in, N * sizeof(int32_t), cudaMemcpyHostToDevice);

  kogge_stone_exclusive_gpu_d(in_d, out_d, N, pos);

  cudaMemcpy(out, out_d, N * sizeof(int32_t), cudaMemcpyDeviceToHost);

  cudaFree(in_d);
  cudaFree(out_d);
}

void radix_gpu(int32_t *in, int32_t *out, size_t N) {
  int32_t *left, *out1, *out2;

  size_t numThreads = BLOCK_DIM;
  size_t numBlocks = cdiv(N, numThreads);

  cudaMallocManaged(&left, N * sizeof(int32_t));
  cudaMallocManaged(&out1, N * sizeof(int32_t));
  cudaMallocManaged(&out2, N * sizeof(int32_t));

  cudaMemcpy(out1, in, N * sizeof(int32_t), cudaMemcpyHostToDevice);

  for (size_t pos = 0; pos < 31; ++pos) {
    kogge_stone_exclusive_gpu(out1, left, N, pos);
    radix_kernel<<<numBlocks, numThreads>>>(out1, left, out2, N, pos);
    cudaDeviceSynchronize();
    int32_t *tmp = out1;
    out1 = out2;
    out2 = tmp;
  }

  cudaMemcpy(out, out1, N * sizeof(int32_t), cudaMemcpyDeviceToHost);

  cudaFree(left);
  cudaFree(out1);
  cudaFree(out2);
}

void print(int32_t *A, size_t N) {
  printf("[\n");
  for (size_t i = 0; i < N; ++i) {
    printf(" %d", A[i]);
  }
  printf("\n]\n");
}

int main() {
  size_t N;
  int32_t *in, *out;

  N = 2049;
  cudaMallocManaged(&in, N * sizeof(int32_t));
  cudaMallocManaged(&out, N * sizeof(int32_t));

  for (size_t i = 0; i < N; ++i) {
    in[i] = rand() % 256;
  }

  radix_gpu(in, out, N);

  for (size_t i = 0; i < N; ++i) {
    printf(" %d", in[i]);
  }
  printf("\n");

  for (size_t i = 0; i < N; ++i) {
    printf(" %d", out[i]);
  }
  printf("\n");

  cudaFree(in);
  cudaFree(out);

  return 0;
}
