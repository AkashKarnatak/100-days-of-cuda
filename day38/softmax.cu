#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__device__ float reduce_max(cg::thread_group g, float *arr) {
  g.sync();
  size_t j = g.thread_rank();
  for (size_t s = g.size() / 2; s > 0; s /= 2) {
    if (j < s) {
      arr[j] = max(arr[j], arr[j + s]);
    }
    g.sync();
  }
  return arr[0];
}

__device__ float reduce_sum(cg::thread_group g, float *arr) {
  g.sync();
  size_t j = g.thread_rank();
  for (size_t s = g.size() / 2; s > 0; s /= 2) {
    if (g.thread_rank() < s) {
      arr[j] += arr[j + s];
    }
    g.sync();
  }
  return arr[0];
}

__global__ void softmax_kernel(float *in, float *out, size_t N, size_t M) {
  size_t tid = threadIdx.x;
  size_t ntid = blockDim.x;
  size_t bid = blockIdx.x;
  auto g = cg::this_thread_block();

  extern __shared__ float sram[];

  float local_sum = 0, local_max = -INFINITY, global_sum, global_max;

  for (size_t i = tid; i < M; i += ntid) {
    float curr = in[bid * M + i];
    if (curr > local_max) {
      local_sum *= expf(local_max - curr);
      local_max = curr;
    }
    local_sum += expf(curr - local_max);
  }

  sram[g.thread_rank()] = local_max;

  global_max = reduce_max(g, sram);

  sram[g.thread_rank()] = local_sum * expf(local_max - global_max);

  global_sum = reduce_sum(g, sram);

  for (size_t i = tid; i < M; i += ntid) {
    out[bid * M + i] = expf(in[bid * M + i] - global_max) / global_sum;
  }
}

extern "C" void softmax(float *in, float *out, size_t N, size_t M) {
  dim3 numThreads(1024);
  dim3 numBlocks(N);
  softmax_kernel<<<numBlocks, numThreads, numThreads.x * sizeof(float)>>>(
      in, out, N, M);
}
