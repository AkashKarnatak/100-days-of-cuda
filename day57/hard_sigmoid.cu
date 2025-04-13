#include <cuda_runtime.h>

__device__ __host__ inline size_t cdiv(size_t a, size_t b) {
  return (a + b - 1) / b;
}

__device__ __forceinline__ float hard_sigmoid(float x) {
  if (x <= -3)
    return 0.0f;
  if (x >= 3)
    return 1.0f;
  return (x + 3.0f) / 6.0f;
}

__global__ void hard_sigmoid_kernel(const float *__restrict__ in,
                                    float *__restrict__ out, size_t N) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N)
    return;
  out[idx] = hard_sigmoid(in[idx]);
}

// Note: input, output are all device pointers to float32 arrays
extern "C" void solution(const float *input, float *output, size_t n,
                         size_t m) {
  size_t N = m, M = n;
  dim3 numThreads(256);
  dim3 numBlocks(cdiv(N * M, numThreads.x));
  hard_sigmoid_kernel<<<numBlocks, numThreads>>>(input, output, N * M);
}
