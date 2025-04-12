#include <cuda_runtime.h>

__device__ __host__ inline size_t cdiv(size_t a, size_t b) {
  return (a + b - 1) / b;
}

__global__ void swish_kernel(const float *__restrict__ in, float *__restrict__ out, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    out[idx] = in[idx] / (1 + expf(-in[idx]));
}

// Note: input, output are all device pointers to float32 arrays
extern "C" void solution(const float* input, float* output, size_t n, size_t m) {
    size_t N = m, M = n;
    dim3 numThreads(256);
    dim3 numBlocks(cdiv(N * M, numThreads.x));
    swish_kernel<<<numBlocks, numThreads>>>(input, output, N * M);
}
