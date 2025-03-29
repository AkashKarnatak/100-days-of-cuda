#include <stdio.h>

inline size_t cdiv(size_t a, size_t b) { return (a + b - 1) / b; }

__device__ __host__ void mergeSeq(float *A, float *B, float *C, size_t N,
                                  size_t M) {
  size_t i = 0, j = 0, k = 0;
  while (i < N && j < M) {
    C[k++] = A[i] < B[j] ? A[i++] : B[j++];
  }
  while (i < N) {
    C[k++] = A[i++];
  }
  while (j < M) {
    C[k++] = B[j++];
  }
}

#define SEQ_LEN 4

__device__ size_t coRank(size_t k, float *A, float *B, float *C, size_t N,
                         size_t M) {
  size_t low = max(0l, k - M);
  size_t high = min(N, k);
  while (true) {
    size_t i = (low + high) / 2;
    size_t j = k - i;
    if (i > 0 && j < M && A[i - 1] > B[j]) {
      high = i;
    } else if (j > 0 && i < N && B[j - 1] > A[i]) {
      low = i;
    } else {
      return i;
    }
  }
}

__global__ void merge_kernel(float *A, float *B, float *C, size_t N, size_t M) {
  size_t k = (blockDim.x * blockIdx.x + threadIdx.x) * SEQ_LEN;

  if (k >= N + M)
    return;

  size_t i = coRank(k, A, B, C, N, M);
  size_t j = k - i;
  size_t kNext = k + SEQ_LEN;
  size_t n = coRank(kNext, A, B, C, N, M) - i;
  size_t m = kNext - n - j;

  mergeSeq(&A[i], &B[j], &C[k], n, m);
}

void merge_cpu(float *A, float *B, float *C, size_t N, size_t M) {
  mergeSeq(A, B, C, N, M);
}

void merge_gpu(float *A, float *B, float *C, size_t N, size_t M) {
  size_t numThreads = 256;
  size_t numBlocks = cdiv(N + M, numThreads * SEQ_LEN);
  merge_kernel<<<numBlocks, numThreads>>>(A, B, C, N, M);
  cudaDeviceSynchronize();
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
  size_t N, M;
  float *A, *B, *C_cpu, *C_gpu;

  N = 3425, M = 454;
  cudaMallocManaged(&A, N * sizeof(float));
  cudaMallocManaged(&B, M * sizeof(float));
  cudaMallocManaged(&C_cpu, M * sizeof(float));
  cudaMallocManaged(&C_gpu, M * sizeof(float));

  for (size_t i = 0; i < N; ++i) {
    A[i] = (float)rand() / RAND_MAX;
  }
  for (size_t i = 0; i < M; ++i) {
    B[i] = (float)rand() / RAND_MAX;
  }

  merge_cpu(A, B, C_cpu, N, M);

  merge_gpu(A, B, C_gpu, N, M);

  printf("Match impl: %s\n", allclose(C_cpu, C_gpu, N) ? "true" : "false");

  cudaFree(A);
  cudaFree(B);
  cudaFree(C_cpu);
  cudaFree(C_gpu);

  return 0;
}
