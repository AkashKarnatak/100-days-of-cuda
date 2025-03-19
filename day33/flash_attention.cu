#include <stdio.h>

size_t cdiv(size_t a, size_t b) { return (a + b - 1) / b; }

__global__ void flash_attention_kernel(float *query, float *key, float *value,
                                       float *output, float *sums, float *maxes,
                                       size_t N, size_t d, size_t Br, size_t Bc,
                                       size_t Tr, size_t Tc) {
  size_t tid = threadIdx.x;
  float *Q, *K, *V, *S;

  extern __shared__ float sram[];
  Q = &sram[0];                   // Br x d
  K = &sram[Br * d];              // Bc x d
  V = &sram[Br * d + Bc * d];     // Bc x d
  S = &sram[Br * d + 2 * Bc * d]; // Br x Bc

  for (size_t j = 0; j < Tc; ++j) {
    __syncthreads();

    // load Kj, Vj
    for (size_t k = 0; k < d; ++k) {
      if (j * Bc + tid < N) {
        K[tid * d + k] = key[(j * Bc + tid) * d + k];
        V[tid * d + k] = value[(j * Bc + tid) * d + k];
      } else {
        K[tid * d + k] = 0;
        V[tid * d + k] = 0;
      }
    }

    __syncthreads();

    for (size_t i = 0; i < Tr; ++i) {
      // load Q
      if (tid < Br) {
        for (size_t k = 0; k < d; ++k) {
          if (i * Br + tid < N) {
            Q[tid * d + k] = query[(i * Br + tid) * d + k];
          } else {
            Q[tid * d + k] = 0;
          }
        }
      }

      __syncthreads();

      if (tid >= Br || i * Br + tid >= N)
        continue;

      float m_prev, l_prev, m_curr, l_curr, m_new, l_new;

      m_prev = maxes[i * Br + tid], l_prev = sums[i * Br + tid];
      m_curr = -INFINITY, l_curr = 0;

      for (size_t col = 0; col < Bc; ++col) {
        float sum = 0.0f;
        for (size_t k = 0; k < d; ++k) {
          sum += Q[tid * d + k] * K[col * d + k];
        }
        if (j * Bc + col < N)
          S[tid * Bc + col] = sum / sqrtf(d);
        else
          S[tid * Bc + col] = -INFINITY;
        m_curr = fmaxf(m_curr, sum / sqrtf(d));
      }

      for (size_t col = 0; col < Bc; ++col) {
        S[tid * Bc + col] = expf(S[tid * Bc + col] - m_curr);
        l_curr += S[tid * Bc + col];
      }

      m_new = max(m_prev, m_curr);
      l_new = l_prev * expf(m_prev - m_new) + l_curr * expf(m_curr - m_new);

      for (size_t col = 0; col < d; ++col) {
        float sum = 0.0f;
        for (size_t k = 0; k < Bc; ++k) {
          sum += S[tid * Bc + k] * V[k * d + col];
        }
        output[(i * Br + tid) * d + col] =
            (l_prev * output[(i * Br + tid) * d + col] * expf(m_prev - m_new) +
             sum * expf(m_curr - m_new)) /
            l_new;
      }
      sums[i * Br + tid] = l_new;
      maxes[i * Br + tid] = m_new;
    }
  }
}

extern "C" {
void flash_attention_gpu(float *query, float *key, float *value, float *output,
                         float *sums, float *maxes, size_t N, size_t d) {
  size_t Br, Bc, Tr, Tc;
  float *query_d, *key_d, *value_d, *output_d, *sums_d, *maxes_d;

  Br = 24, Bc = 24;
  Tr = cdiv(N, Br), Tc = cdiv(N, Bc);

  cudaMalloc(&query_d, (N * d) * sizeof(float));
  cudaMalloc(&key_d, (N * d) * sizeof(float));
  cudaMalloc(&value_d, (N * d) * sizeof(float));
  cudaMalloc(&output_d, (N * d) * sizeof(float));
  cudaMalloc(&sums_d, N * sizeof(float));
  cudaMalloc(&maxes_d, N * sizeof(float));

  cudaMemcpy(query_d, query, (N * d) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(key_d, key, (N * d) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(value_d, value, (N * d) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(output_d, output, (N * d) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(sums_d, sums, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(maxes_d, maxes, N * sizeof(float), cudaMemcpyHostToDevice);

  dim3 numThreads(Bc);
  dim3 numBlocks(1);
  size_t sramMem = (Br * d    // Q
                    + Bc * d  // K
                    + Bc * d  // V
                    + Br * Bc // S
                    ) *
                   sizeof(float);
  flash_attention_kernel<<<numBlocks, numThreads, sramMem>>>(
      query_d, key_d, value_d, output_d, sums_d, maxes_d, N, d, Br, Bc, Tr, Tc);
  cudaError_t err = cudaGetLastError(); // Check for launch errors
  if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
  }
  cudaDeviceSynchronize();

  cudaMemcpy(output, output_d, (N * d) * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(sums, sums_d, N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(maxes, maxes_d, N * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(query_d);
  cudaFree(key_d);
  cudaFree(value_d);
  cudaFree(output_d);
  cudaFree(sums_d);
  cudaFree(maxes_d);
}
}

int main() {
  cudaDeviceSynchronize();

  size_t N, d;
  float *query, *key, *value, *output, *sums, *maxes;

  N = 64, d = 64;

  query = (float *)malloc((N * d) * sizeof(float));
  key = (float *)malloc((N * d) * sizeof(float));
  value = (float *)malloc((N * d) * sizeof(float));
  output = (float *)malloc((N * d) * sizeof(float));
  sums = (float *)malloc(N * sizeof(float));
  maxes = (float *)malloc(N * sizeof(float));

  for (size_t i = 0; i < N * d; ++i) {
    query[i] = (float)rand() / RAND_MAX;
    key[i] = (float)rand() / RAND_MAX;
    value[i] = (float)rand() / RAND_MAX;
    output[i] = 0;
  }

  for (size_t i = 0; i < N; ++i) {
    sums[i] = 0;
    maxes[i] = -INFINITY;
  }

  flash_attention_gpu(query, key, value, output, sums, maxes, N, d);

  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < d; ++j) {
      printf(" %.4f", output[i * d + j]);
    }
    printf("\n");
  }

  free(query);
  free(key);
  free(value);
  free(output);
  free(sums);
  free(maxes);
}
