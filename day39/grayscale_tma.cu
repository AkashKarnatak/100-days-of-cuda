#include <cuda/barrier>
#include <cuda/ptx>
using barrier = cuda::barrier<cuda::thread_scope_block>;

inline unsigned int cdiv(unsigned int a, unsigned int b) {
  return (a + b - 1) / b;
}

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
static constexpr size_t buf_len = 256 * 3;
__global__ void grayscale_kernel(float *__restrict__ in,
                                 float *__restrict__ out) {
  __shared__ alignas(16) float smem_data[buf_len];
  size_t tid = threadIdx.x;
  size_t offset = blockIdx.x * blockDim.x;

#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier bar;
  if (threadIdx.x == 0) {
    init(&bar, blockDim.x); // a)
    asm volatile("fence.proxy.async.shared::cta;");
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    cuda::memcpy_async(smem_data, in + offset * 3,
                       cuda::aligned_size_t<16>(sizeof(smem_data)), bar);
  }
  barrier::arrival_token token = bar.arrive();

  bar.wait(std::move(token));

  float gray = __fmaf_rn(0.1140f, smem_data[tid * 3 + 2],
                         __fmaf_rn(0.5870f, smem_data[tid * 3 + 1],
                                   __fmul_rn(0.2989f, smem_data[tid * 3])));
  __syncthreads();
  smem_data[tid] = gray;

  asm volatile("fence.proxy.async.shared::cta;");
  __syncthreads();

  if (threadIdx.x == 0) {
    asm volatile(
        "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(
            out + offset),
        "l"(smem_data), "r"(1024));
    asm volatile("cp.async.bulk.commit_group;");
    asm volatile("cp.async.bulk.wait_group.read 0;");
  }
}

int main() {
  float *in, *out, *in_d, *in2_d, *out_d;
  size_t N, M;

  N = M = 8192;

  in = (float *)malloc(N * M * 4 * sizeof(float));
  out = (float *)malloc(N * M * sizeof(float));

  cudaMalloc(&in_d, N * M * 4 * sizeof(float));
  cudaMalloc(&in2_d, N * M * 4 * sizeof(float));
  cudaMalloc(&out_d, N * M * sizeof(float));

  cudaMemcpy(in_d, in, N * M * 4 * sizeof(float), cudaMemcpyHostToDevice);

  dim3 numThreads(256);
  dim3 numBlocks(cdiv(N * M, numThreads.x * 4));

  cudaDeviceSynchronize();
  start_timer(&t);
  grayscale_kernel<<<numBlocks, numThreads>>>(in_d, out_d);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError(); // Check for launch errors
  if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
  }
  stop_timer(&t);
  printf("Time diff: %.3f ms\n", time_diff(&t) * 1000);

  cudaMemcpy(out, out_d, N * M * sizeof(float), cudaMemcpyHostToDevice);

  cudaFree(in_d);
  cudaFree(out_d);

  free(in);
  free(out);
}
