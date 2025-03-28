#include <stdint.h>
#include <stdio.h>

inline size_t cdiv(size_t a, size_t b) { return (a + b - 1) / b; }

__global__ void histogram_kernel(uint8_t *in, uint32_t *bins, size_t N) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ uint32_t sram[256];

  if (i >= N)
    return;

  if (threadIdx.x < 256) {
    sram[threadIdx.x] = 0;
  }

  __syncthreads();

  atomicAdd(&sram[in[i]], 1);

  __syncthreads();

  if (threadIdx.x < 256) {
    atomicAdd(&bins[threadIdx.x], sram[threadIdx.x]);
  }
}

void histogram_gpu(uint8_t *img, uint32_t *bins, size_t N) {
  size_t numThreads = 256;
  size_t numBlocks = cdiv(N, numThreads);
  histogram_kernel<<<numBlocks, numThreads>>>(img, bins, N);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
    exit(1);
  }
}

void histogram_cpu(uint8_t *img, uint32_t *bins, size_t N) {
  for (size_t i = 0; i < N; ++i) {
    ++bins[img[i]];
  }
}

bool allclose(uint32_t *A, uint32_t *B, size_t N) {
  for (size_t i = 0; i < N; ++i) {
    if (A[i] != B[i]) {
      printf("Mismatch at (%ld,): A = %u and B = %u\n", i, A[i], B[i]);
      return false;
    }
  }
  return true;
}

void print(uint32_t *A, size_t N) {
  printf("[\n");
  for (size_t i = 0; i < N; ++i) {
    printf(" %u", A[i]);
  }
  printf("\n]\n");
}

int main() {
  size_t N, M;
  uint8_t *img;
  uint32_t *bins_cpu, *bins_gpu;

  N = M = 1024;

  cudaMallocManaged(&img, N * M * sizeof(uint8_t));
  cudaMallocManaged(&bins_cpu, 256 * sizeof(uint32_t));
  cudaMallocManaged(&bins_gpu, 256 * sizeof(uint32_t));

  cudaMemset(bins_cpu, 0, 256 * sizeof(uint32_t));
  cudaMemset(bins_gpu, 0, 256 * sizeof(uint32_t));

  for (size_t i = 0; i < N * M; ++i) {
    img[i] = rand() % 256;
  }

  histogram_cpu(img, bins_cpu, N * M);

  histogram_gpu(img, bins_gpu, N * M);

  printf("Match impl: %s\n",
         allclose(bins_cpu, bins_gpu, 256) ? "true" : "false");

  cudaFree(img);
  cudaFree(bins_cpu);
  cudaFree(bins_gpu);
}
