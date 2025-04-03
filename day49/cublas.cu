#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define N 3
#define M 4
#define K 2

void printMatrix(const float *A, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      printf("%15.5f", A[i * cols + j]);
    }
    printf("\n");
  }
}

int main() {
  float A[N * K] = {1, 2, 3, 4, 5, 6};
  float B[K * M] = {7, 8, 9, 10, 11, 12, 13, 14};
  float C[N * M];

  float *A_d, *B_d, *C_d;
  cublasHandle_t handle;
  cublasCreate(&handle);

  cudaMalloc(&A_d, N * K * sizeof(float));
  cudaMalloc(&B_d, K * M * sizeof(float));
  cudaMalloc(&C_d, N * M * sizeof(float));

  cublasSetMatrix(N, K, sizeof(float), A, N, A_d, N);
  cublasSetMatrix(K, M, sizeof(float), B, K, B_d, K);

  const float alpha = 1.0f, beta = 0.0f;
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, B_d, M, A_d, K,
              &beta, C_d, M);

  cublasGetMatrix(N, M, sizeof(float), C_d, N, C, N);

  printf("Result matrix:\n");
  printMatrix(C, N, M);

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
  cublasDestroy(handle);

  return 0;
}
