#include <hip/hip_runtime.h>
#include <stdio.h>

inline size_t cdiv(size_t a, size_t b) { return (a + b - 1) / b; }

#define HIP_CHECK(call)                                                        \
  do {                                                                         \
    hipError_t err = call;                                                     \
    if (err != hipSuccess) {                                                   \
      fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__,          \
              hipGetErrorString(err));                                         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

__global__ void vec_add_kernel(float *__restrict__ A, float *__restrict__ B,
                               float *__restrict__ C, size_t N) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < N) {
    C[idx] = A[idx] + B[idx];
  }
}

void vec_add_cpu(float *A, float *B, float *C, size_t N) {
  for (size_t i = 0; i < N; ++i) {
    C[i] = A[i] + B[i];
  }
}

void print(float *A, size_t N) {
  for (size_t i = 0; i < N; ++i) {
    printf("%12.4f", A[i]);
  }
  printf("\n");
}

bool allclose(float *A, float *B, size_t N) {
  for (size_t i = 0; i < N; ++i) {
    if (abs(A[i] - B[i]) > 1e-5) {
      printf("Mismatch at %lu, A = %f and B = %f\n", i, A[i], B[i]);
      return false;
    }
  }
  return true;
}

int main() {
  size_t N;
  float *A, *B, *C, *C_cpu;

  N = 1'000'000;

  HIP_CHECK(hipMallocManaged(&A, N * sizeof(float)));
  HIP_CHECK(hipMallocManaged(&B, N * sizeof(float)));
  HIP_CHECK(hipMallocManaged(&C, N * sizeof(float)));
  HIP_CHECK(hipMallocManaged(&C_cpu, N * sizeof(float)));

  for (size_t i = 0; i < N; ++i) {
    A[i] = (float)rand() / RAND_MAX;
    B[i] = (float)rand() / RAND_MAX;
  }

  vec_add_cpu(A, B, C_cpu, N);

  dim3 numThreads(1024);
  dim3 numBlocks(cdiv(N, numThreads.x));
  hipLaunchKernelGGL(vec_add_kernel, numBlocks, numThreads, 0, 0, A, B, C, N);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  // print(C, N);
  // print(C_cpu, N);

  printf("Match impl: %s\n", allclose(C_cpu, C, N) ? "true" : "false");

  HIP_CHECK(hipFree(A));
  HIP_CHECK(hipFree(B));

  return 0;
}
