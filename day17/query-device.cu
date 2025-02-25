#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

int main() {
  int32_t devCount;
  cudaDeviceProp devProp;

  cudaGetDeviceCount(&devCount);

  printf("%d\n", devCount);

  for (size_t i = 0; i < (size_t)devCount; ++i) {
    cudaGetDeviceProperties(&devProp, i);
    printf("Number of SMs: %d\n", devProp.multiProcessorCount);
    printf("Max threads per block: %d\n", devProp.maxThreadsPerBlock);
    printf("Max blocks per SM: %d\n", devProp.maxBlocksPerMultiProcessor);
    printf("Max threads per SM: %d\n", devProp.maxThreadsPerMultiProcessor);
    printf("Max registers per SM: %d\n", devProp.maxThreadsPerMultiProcessor);
    printf("Warp size: %d\n", devProp.warpSize);
    printf("Clock rate: %d\n", devProp.clockRate);
    printf("Max threads for dim 0: %d\n", devProp.maxThreadsDim[0]);
    printf("Max threads for dim 1: %d\n", devProp.maxThreadsDim[1]);
    printf("Max threads for dim 2: %d\n", devProp.maxThreadsDim[2]);
    printf("Max number of block for dim 0: %d\n", devProp.maxGridSize[0]);
    printf("Max number of block for dim 1: %d\n", devProp.maxGridSize[1]);
    printf("Max number of block for dim 2: %d\n", devProp.maxGridSize[2]);
  }

  return 0;
}
