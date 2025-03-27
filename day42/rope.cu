#include <stdio.h>

inline size_t cdiv(size_t a, size_t b) { return (a + b - 1) / b; }

__global__ void rope_kernel(float *out, float *in, size_t N, size_t T,
                            size_t d) {
  size_t i = blockIdx.x;
  size_t j = (blockDim.y * blockIdx.y + threadIdx.y) * 2;

  if (j + 1 >= d)
    return;

  size_t pos = i % T;
  float theta = pos * expf(j * -logf(10000) / d);
  float2 in2 = *(float2 *)&in[i * d + j];
  float2 out2;

  out2.x = in2.x * cosf(theta) - in2.y * sinf(theta);
  out2.y = in2.x * sinf(theta) + in2.y * cosf(theta);

  *(float2 *)&out[i * d + j] = out2;
}

extern "C" void rope_gpu(float *out, float *in, size_t N, size_t T, size_t d) {
  dim3 numThreads(1, 256);
  dim3 numBlocks(N, cdiv(d, numThreads.y * 2));

  rope_kernel<<<numBlocks, numThreads>>>(out, in, N, T, d);
  cudaDeviceSynchronize();

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
  }
}
