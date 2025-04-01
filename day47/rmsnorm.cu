#define BLOCK_DIM 1024

__device__ __forceinline__ float warpReduceSum(float val) {
#pragma unroll
  for (size_t s = warpSize / 2; s > 0; s /= 2) {
    val += __shfl_down_sync(0xffffffff, val, s);
  }
  return val;
}

__global__ void rmsnorm_kernel(float *in, float *out, size_t B, size_t N) {
  size_t tid = threadIdx.x;
  size_t row = blockIdx.x;

  float local_sq = 0;

  for (size_t i = tid; i < N; i += blockDim.x) {
    local_sq += in[row * N + i] * in[row * N + i];
  }

  local_sq = warpReduceSum(local_sq);

  size_t warpIdx = tid / warpSize;
  size_t laneIdx = tid % warpSize;

  __shared__ float sram[32];

  if (laneIdx == 0) {
    sram[warpIdx] = local_sq;
  }
  if (tid < warpSize && tid >= (blockDim.x / warpSize)) {
    sram[tid] = 0;
  }
  __syncthreads();

  if (tid < warpSize) {
    sram[tid] = warpReduceSum(sram[tid]);
  }
  __syncthreads();

  for (size_t i = tid; i < N; i += blockDim.x) {
    out[row * N + i] = in[row * N + i] / sqrt(sram[0] / N + 1e-5);
  }
}

extern "C" void rmsnorm_gpu(float *in, float *out, size_t B, size_t N) {
  rmsnorm_kernel<<<B, BLOCK_DIM>>>(in, out, B, N);
  cudaDeviceSynchronize();
}
