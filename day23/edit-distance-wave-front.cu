#include <cassert>
#include <cstddef>
#include <cstdint>
#include <stdio.h>

#define SWAP(a, b, T)                                                          \
  do {                                                                         \
    T tmp = a;                                                                 \
    a = b;                                                                     \
    b = tmp;                                                                   \
  } while (0)

int32_t min(int32_t a, int32_t b);

__global__ void wagner_fischer_kernel(int32_t *__restrict__ dist,
                                      char *__restrict__ s1,
                                      char *__restrict__ s2, size_t n1,
                                      size_t n2) {
  size_t j = threadIdx.x;

  for (size_t wave = 0; wave <= (n1 + n2); ++wave) {
    int32_t i = wave - j;

    if (i >= 0 && i <= n1) {
      if (i == 0) {
        dist[j] = j;
        continue;
      }

      if (j == 0) {
        dist[i * (n2 + 1)] = i;
        continue;
      }

      if (s1[i - 1] == s2[j - 1]) {
        dist[i * (n2 + 1) + j] = dist[(i - 1) * (n2 + 1) + (j - 1)];
      } else {
        dist[i * (n2 + 1) + j] = min(min(dist[(i - 1) * (n2 + 1) + (j - 1)],
                                         dist[(i - 1) * (n2 + 1) + j]),
                                     dist[i * (n2 + 1) + (j - 1)]) +
                                 1;
      }
    }
    __syncthreads();
  }
}

void wagner_fischer_gpu(int32_t *dist, char *__restrict__ s1,
                        char *__restrict__ s2, size_t n1, size_t n2) {
  int32_t *dist_d;
  char *s1_d, *s2_d;
  cudaMalloc(&dist_d, (n1 + 1) * (n2 + 1) * sizeof(int32_t));
  cudaMalloc(&s1_d, n1 * sizeof(char));
  cudaMalloc(&s2_d, n2 * sizeof(char));

  cudaMemcpy(s1_d, s1, n1 * sizeof(char), cudaMemcpyHostToDevice);
  cudaMemcpy(s2_d, s2, n2 * sizeof(char), cudaMemcpyHostToDevice);

  dim3 numThreads(n2 + 1);
  dim3 numBlocks(1);
  wagner_fischer_kernel<<<numBlocks, numThreads>>>(dist_d, s1_d, s2_d, n1, n2);

  cudaMemcpy(dist, dist_d, (n1 + 1) * (n2 + 1) * sizeof(int32_t),
             cudaMemcpyDeviceToHost);

  cudaFree(dist_d);
  cudaFree(s1_d);
  cudaFree(s2_d);
}

void wagner_fischer_cpu(int32_t *dist, char *__restrict__ s1,
                        char *__restrict__ s2, size_t n1, size_t n2) {
  for (size_t i = 0; i <= n1; ++i) {
    dist[i * (n2 + 1)] = i; // j = 0
  }
  for (size_t j = 0; j <= n2; ++j) {
    dist[j] = j; // i = 0
  }
  for (size_t i = 1; i <= n1; ++i) {
    for (size_t j = 1; j <= n2; ++j) {
      if (s1[i - 1] == s2[j - 1]) {
        dist[i * (n2 + 1) + j] = dist[(i - 1) * (n2 + 1) + (j - 1)];
      } else {
        dist[i * (n2 + 1) + j] = min(min(dist[(i - 1) * (n2 + 1) + (j - 1)],
                                         dist[(i - 1) * (n2 + 1) + j]),
                                     dist[i * (n2 + 1) + (j - 1)]) +
                                 1;
      }
    }
  }
}

void print(int32_t *A, size_t n1, size_t n2) {
  printf("[\n");
  for (size_t i = 0; i <= n1; ++i) {
    for (size_t j = 0; j <= n2; ++j) {
      printf(" %d", A[i * (n2 + 1) + j]);
    }
    printf("\n");
  }
  printf("\n]\n");
}

bool allclose(int32_t *A, int32_t *B, size_t n1, size_t n2) {
  for (size_t i = 0; i <= n1; ++i) {
    for (size_t j = 0; j <= n2; ++j) {
      if (A[i * (n2 + 1) + j] != B[i * (n2 + 1) + j]) {
        printf("Mismatch at (%lu, %lu), A = %d and B = %d\n", i, j,
               A[i * (n2 + 1) + j], B[i * (n2 + 1) + j]);
        return false;
      }
    }
  }
  return true;
}

int main() {
  char str1[] = "card";
  char str2[] = "curl";

  char *s1 = str1, *s2 = str2;
  size_t n1 = strlen(s1);
  size_t n2 = strlen(s2);

  if (n2 > n1) {
    SWAP(s1, s2, char *);
    SWAP(n1, n2, size_t);
  }

  int32_t *dist_cpu = (int32_t *)malloc((n1 + 1) * (n2 + 1) * sizeof(int32_t));
  assert(dist_cpu != NULL);
  int32_t *dist_gpu = (int32_t *)malloc((n1 + 1) * (n2 + 1) * sizeof(int32_t));
  assert(dist_gpu != NULL);

  wagner_fischer_cpu(dist_cpu, s1, s2, n1, n2);

  wagner_fischer_gpu(dist_gpu, s1, s2, n1, n2);

  print(dist_cpu, n1, n2);
  print(dist_gpu, n1, n2);

  printf("GPU and CPU impl match: %s\n",
         allclose(dist_cpu, dist_gpu, n1, n2) ? "true" : "false");

  free(dist_cpu);
  free(dist_gpu);
}
