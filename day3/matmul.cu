#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

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
struct Mat {
  float *data;
  int32_t w;
  int32_t h;
};

__host__ __device__ float mat_at(struct Mat *m, int32_t i, int32_t j) {
  return m->data[i * m->w + j];
}

__host__ __device__ void mat_set(struct Mat *m, int32_t i, int32_t j, float x) {
  m->data[i * m->w + j] = x;
}

__host__ __device__ void mat_update(struct Mat *m, int32_t i, int32_t j,
                                    float x) {
  m->data[i * m->w + j] += x;
}

__host__ __device__ void mat_free(struct Mat *m) { free(m->data); }

__host__ __device__ void mat_print(struct Mat *m) {
  printf("[\n");
  for (int32_t i = 0; i < m->h; ++i) {
    for (int32_t j = 0; j < m->w; ++j) {
      printf(" %f", mat_at(m, i, j));
    }
    printf("\n");
  }
  printf("]\n");
}

__host__ __device__ bool mat_equal(struct Mat *A, struct Mat *B) {
  if (A->h != B->h || A->w != B->w)
    return false;

  for (int32_t i = 0; i < A->h; ++i) {
    for (int32_t j = 0; j < A->w; ++j) {
      if (abs(mat_at(A, i, j) - mat_at(B, i, j)) > 1e-4)
        return false;
    }
  }
  return true;
}

int32_t cdiv(int32_t a, int32_t b) { return (a + b - 1) / b; }

void matmul_cpu(struct Mat A, struct Mat B, struct Mat C) {
  for (int32_t i = 0; i < C.h; ++i) {
    for (int32_t j = 0; j < C.w; ++j) {
      mat_set(&C, i, j, 0);
      for (int32_t k = 0; k < A.w; ++k) {
        mat_update(&C, i, j, mat_at(&A, i, k) * mat_at(&B, k, j));
      }
    }
  }
}

__global__ void matmul_kernel(struct Mat A, struct Mat B, struct Mat C) {
  int32_t row = blockIdx.y * blockDim.y + threadIdx.y;
  int32_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= C.h || col >= C.w)
    return;
  float sum = 0;
  for (int32_t k = 0; k < A.w; ++k) {
    sum += A.data[row * A.w + k] * B.data[k * B.w + col];
  }
  C.data[row * C.w + col] = sum;
}

__global__ void matmul_row_kernel(struct Mat A, struct Mat B, struct Mat C) {
  int32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= C.h)
    return;
  for (int32_t col = 0; col < C.w; ++col) {
    float sum = 0;
    for (int32_t k = 0; k < A.w; ++k) {
      sum += A.data[row * A.w + k] * B.data[k * B.w + col];
    }
    C.data[row * C.w + col] = sum;
  }
}

__global__ void matmul_col_kernel(struct Mat A, struct Mat B, struct Mat C) {
  int32_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= C.w)
    return;
  for (int32_t row = 0; row < C.h; ++row) {
    float sum = 0;
    for (int32_t k = 0; k < A.w; ++k) {
      sum += A.data[row * A.w + k] * B.data[k * B.w + col];
    }
    C.data[row * C.w + col] = sum;
  }
}

void matmul_gpu(struct Mat A, struct Mat B, struct Mat C) {
  struct Mat A_d, B_d, C_d;

  // allocate memory on GPU
  A_d = A, B_d = B, C_d = C;
  cudaMalloc(&A_d.data, (A_d.h * A_d.w) * sizeof(float));
  cudaMalloc(&B_d.data, (B_d.h * B_d.w) * sizeof(float));
  cudaMalloc(&C_d.data, (C_d.h * C_d.w) * sizeof(float));

  // copy data to device
  cudaMemcpy(A_d.data, A.data, (A_d.h * A_d.w) * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(B_d.data, B.data, (B_d.h * B_d.w) * sizeof(float),
             cudaMemcpyHostToDevice);

  // launch kernel
  dim3 numThreads(32, 32);
  dim3 numBlocks(cdiv(C.h, numThreads.x), cdiv(C.w, numThreads.y));
  matmul_kernel<<<numBlocks, numThreads>>>(A_d, B_d, C_d);

  // copy data back to CPU
  cudaMemcpy(C.data, C_d.data, (C_d.h * C_d.w) * sizeof(float),
             cudaMemcpyDeviceToHost);

  // free resource
  cudaFree(A_d.data);
  cudaFree(B_d.data);
  cudaFree(C_d.data);
}

void matmul_row_gpu(struct Mat A, struct Mat B, struct Mat C) {
  struct Mat A_d, B_d, C_d;

  // allocate memory on GPU
  A_d = A, B_d = B, C_d = C;
  cudaMalloc(&A_d.data, (A_d.h * A_d.w) * sizeof(float));
  cudaMalloc(&B_d.data, (B_d.h * B_d.w) * sizeof(float));
  cudaMalloc(&C_d.data, (C_d.h * C_d.w) * sizeof(float));

  // copy data to device
  cudaMemcpy(A_d.data, A.data, (A_d.h * A_d.w) * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(B_d.data, B.data, (B_d.h * B_d.w) * sizeof(float),
             cudaMemcpyHostToDevice);

  // launch kernel
  int32_t numThreads = 1024;
  int32_t numBlocks = cdiv(C.h, numThreads);
  matmul_row_kernel<<<numBlocks, numThreads>>>(A_d, B_d, C_d);

  // copy data back to CPU
  cudaMemcpy(C.data, C_d.data, (C_d.h * C_d.w) * sizeof(float),
             cudaMemcpyDeviceToHost);

  // free resource
  cudaFree(A_d.data);
  cudaFree(B_d.data);
  cudaFree(C_d.data);
}

void matmul_col_gpu(struct Mat A, struct Mat B, struct Mat C) {
  struct Mat A_d, B_d, C_d;

  // allocate memory on GPU
  A_d = A, B_d = B, C_d = C;
  cudaMalloc(&A_d.data, (A_d.h * A_d.w) * sizeof(float));
  cudaMalloc(&B_d.data, (B_d.h * B_d.w) * sizeof(float));
  cudaMalloc(&C_d.data, (C_d.h * C_d.w) * sizeof(float));

  // copy data to device
  cudaMemcpy(A_d.data, A.data, (A_d.h * A_d.w) * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(B_d.data, B.data, (B_d.h * B_d.w) * sizeof(float),
             cudaMemcpyHostToDevice);

  // launch kernel
  int32_t numThreads = 1024;
  int32_t numBlocks = cdiv(C.w, numThreads);
  matmul_col_kernel<<<numBlocks, numThreads>>>(A_d, B_d, C_d);

  // copy data back to CPU
  cudaMemcpy(C.data, C_d.data, (C_d.h * C_d.w) * sizeof(float),
             cudaMemcpyDeviceToHost);

  // free resource
  cudaFree(A_d.data);
  cudaFree(B_d.data);
  cudaFree(C_d.data);
}

int32_t main() {
  cudaDeviceSynchronize();

  int32_t p, q, r;
  struct Mat A, B, C, D;

  // allocate two matrices of dim pxq and qxr
  p = 1024, q = 1024, r = 1024;
  A.h = p, A.w = q;
  B.h = q, B.w = r;
  C.h = p, C.w = r;
  D.h = p, D.w = r;
  A.data = (float *)malloc((A.h * A.w) * sizeof(float));
  B.data = (float *)malloc((B.h * B.w) * sizeof(float));
  C.data = (float *)malloc((C.h * C.w) * sizeof(float));
  D.data = (float *)malloc((D.h * D.w) * sizeof(float));
  for (int32_t i = 0; i < p * q; ++i) {
    A.data[i] = (float)rand() / RAND_MAX;
  }
  for (int32_t i = 0; i < q * r; ++i) {
    B.data[i] = (float)rand() / RAND_MAX;
  }

  // mat_print(&A);
  // mat_print(&B);
  start_timer(&t);
  matmul_cpu(A, B, C);
  stop_timer(&t);
  printf("CPU time: %f\n", time_diff(&t));
  // mat_print(&C);

  start_timer(&t);
  matmul_gpu(A, B, D);
  cudaDeviceSynchronize();
  stop_timer(&t);
  printf("GPU time(each cell): %f\n", time_diff(&t));
  // mat_print(&D);

  start_timer(&t);
  matmul_row_gpu(A, B, D);
  cudaDeviceSynchronize();
  stop_timer(&t);
  printf("GPU time(each row): %f\n", time_diff(&t));
  // mat_print(&D);

  start_timer(&t);
  matmul_col_gpu(A, B, D);
  cudaDeviceSynchronize();
  stop_timer(&t);
  printf("GPU time(each col): %f\n", time_diff(&t));
  // mat_print(&D);

  printf("Match: %s\n", mat_equal(&C, &D) ? "true" : "false");

  mat_free(&A);
  mat_free(&B);
  mat_free(&C);
  mat_free(&D);
}
