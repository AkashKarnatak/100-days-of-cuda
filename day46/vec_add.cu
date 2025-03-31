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
inline size_t cdiv(size_t a, size_t b) { return (a + b - 1) / b; }

__global__ void vec_add_kernel(float *A, float *B, float *C, size_t N) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N)
    return;

  C[idx] = A[idx] + B[idx];
}

void vec_add_gpu(float *A, float *B, float *C, size_t N) {
  cudaDeviceSynchronize();

  float *A_d, *B_d, *C_d;

  cudaMalloc(&A_d, N * sizeof(float));
  cudaMalloc(&B_d, N * sizeof(float));
  cudaMalloc(&C_d, N * sizeof(float));

  start_timer(&t);

  cudaMemcpy(A_d, A, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B, N * sizeof(float), cudaMemcpyHostToDevice);

  size_t numThreads = 256;
  size_t numBlocks = cdiv(N, 256);
  vec_add_kernel<<<numBlocks, numThreads>>>(A_d, B_d, C_d, N);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
    exit(1);
  }

  cudaMemcpy(C, C_d, N * sizeof(float), cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  stop_timer(&t);
  printf("GPU time: %f\n", time_diff(&t));

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
}

void vec_add_pipelined_gpu(float *A, float *B, float *C, size_t N) {
  cudaDeviceSynchronize();

  float *A_d, *B_d, *C_d;

  cudaMalloc(&A_d, N * sizeof(float));
  cudaMalloc(&B_d, N * sizeof(float));
  cudaMalloc(&C_d, N * sizeof(float));

  size_t numStreams = 8;
  size_t segmentLen = cdiv(N, numStreams);
  cudaStream_t streams[numStreams];

  for (size_t i = 0; i < numStreams; ++i) {
    cudaStreamCreate(&streams[i]);
  }

  start_timer(&t);

  for (size_t i = 0; i < numStreams; ++i) {
    size_t start = i * segmentLen;
    size_t Nsegment = (N - start) < segmentLen ? (N - start) : segmentLen;

    cudaMemcpyAsync(&A_d[start], &A[start], Nsegment * sizeof(float),
                    cudaMemcpyHostToDevice, streams[i]);
    cudaMemcpyAsync(&B_d[start], &B[start], Nsegment * sizeof(float),
                    cudaMemcpyHostToDevice, streams[i]);

    size_t numThreads = 256;
    size_t numBlocks = cdiv(Nsegment, numThreads);
    vec_add_kernel<<<numBlocks, numThreads, 0, streams[i]>>>(
        &A_d[start], &B_d[start], &C_d[start], Nsegment);

    cudaMemcpyAsync(&C[start], &C_d[start], Nsegment * sizeof(float),
                    cudaMemcpyDeviceToHost, streams[i]);
  }

  cudaDeviceSynchronize();

  stop_timer(&t);
  printf("GPU time (pipelined): %f\n", time_diff(&t));

  for (size_t i = 0; i < numStreams; ++i) {
    cudaStreamDestroy(streams[i]);
  }

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
}

void vec_add_cpu(float *A, float *B, float *C, size_t N) {
  start_timer(&t);
  for (size_t i = 0; i < N; ++i) {
    C[i] = A[i] + B[i];
  }
  stop_timer(&t);
  printf("CPU time: %f\n", time_diff(&t));
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

void print(float *A, size_t N) {
  for (size_t i = 0; i < N; ++i) {
    printf(" %f", A[i]);
  }
  printf("\n");
}

int main() {
  size_t N;
  float *A, *B, *C, *C_gpu;

  N = 100'000'000;
  cudaMallocHost(&A, N * sizeof(float));
  cudaMallocHost(&B, N * sizeof(float));
  cudaMallocHost(&C, N * sizeof(float));
  cudaMallocHost(&C_gpu, N * sizeof(float));

  for (size_t i = 0; i < N; ++i) {
    A[i] = (float)rand() / RAND_MAX;
    B[i] = (float)rand() / RAND_MAX;
  }

  vec_add_cpu(A, B, C, N);

  // vec_add_gpu(A, B, C_gpu, N);
  // printf("Match impl: %s\n", allclose(C, C_gpu, N) ? "true" : "false");

  vec_add_pipelined_gpu(A, B, C_gpu, N);
  printf("Match impl (pipelined): %s\n",
         allclose(C, C_gpu, N) ? "true" : "false");

  cudaFreeHost(A);
  cudaFreeHost(B);
  cudaFreeHost(C);
  cudaFreeHost(C_gpu);
}
