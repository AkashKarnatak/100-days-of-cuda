#include "common.h"
#include <stdio.h>

const size_t BUFFER_SIZE = 100;
const size_t BLUR_RADIUS = 1;

struct timer t;

__host__ __device__ struct pixel blur(uint32_t row, uint32_t col,
                                      struct pixel *img, uint32_t width,
                                      uint32_t height) {
  uint32_t r = 0, g = 0, b = 0;
  for (int32_t inRow = row - BLUR_RADIUS; inRow <= row + BLUR_RADIUS; ++inRow) {
    for (int32_t inCol = col - BLUR_RADIUS; inCol <= col + BLUR_RADIUS;
         ++inCol) {
      if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
        uint32_t idx = inRow * width + inCol;
        r += img[idx].r, g += img[idx].g, b += img[idx].b;
      }
    }
  }
  uint32_t D = (2 * BLUR_RADIUS + 1) * (2 * BLUR_RADIUS + 1);
  r /= D, g /= D, b /= D;
  return {.r = (uint8_t)r, .g = (uint8_t)g, .b = (uint8_t)b};
}

void blur_cpu(struct pixel *img, struct pixel *blur_img, uint32_t width,
              uint32_t height) {
  for (uint32_t i = 0; i < (width * height); ++i) {
    uint32_t col = i % width;
    uint32_t row = i - col * width;
    blur_img[i] = blur(row, col, img, width, height);
  }
}

__global__ void blur_kernel(struct pixel *img, struct pixel *blur_img,
                            uint32_t width, uint32_t height) {
  uint32_t row = blockDim.y * blockIdx.y + threadIdx.y;
  uint32_t col = blockDim.x * blockIdx.x + threadIdx.x;
  if (row >= height || col >= width)
    return;
  uint32_t i = height * row + col;
  blur_img[i] = blur(row, col, img, width, height);
}

void blur_gpu(struct pixel *img, struct pixel *blur_img, uint32_t width,
              uint32_t height) {
  struct pixel *img_d;
  uint8_t *gray_img_d;

  // allocate memory on GPU
  start_timer(&t);
  cudaMalloc(&img_d, (width * height) * sizeof(struct pixel));
  cudaMalloc(&gray_img_d, (width * height) * sizeof(uint8_t));

  // copy data from host to device
  cudaMemcpy(img_d, img, (width * height) * sizeof(struct pixel),
             cudaMemcpyHostToDevice);
  stop_timer(&t);
  printf("CPU to GPU copy time: %f\n", time_diff(&t));

  // perform computation
  dim3 numThreads(32, 32);
  dim3 numBlocks((width + numThreads.x - 1) / numThreads.x,
                 (height + numThreads.y - 1) / numThreads.y);
  blur_kernel<<<numBlocks, numThreads>>>(img, blur_img, width, height);

  // copy data from device to host
  cudaMemcpy(blur_img, gray_img_d, (width * height) * sizeof(uint8_t),
             cudaMemcpyDeviceToHost);

  // free memory
  cudaFree(img_d);
  cudaFree(gray_img_d);
}

int32_t main() {
  cudaDeviceSynchronize();

  char buf[BUFFER_SIZE];
  uint32_t width, height;
  struct pixel *img, *blur_img;

  printf("=========== BLUR ===========\n");

  // read image
  width = 10240, height = 10240;
  img = (struct pixel *)malloc((width * height) * sizeof(struct pixel));
  blur_img = (struct pixel *)malloc((width * height) * sizeof(struct pixel));
  snprintf(buf, BUFFER_SIZE, "__artifacts__/rgb_image_%dx%d", width, height);
  read_image(buf, img, width, height);

  // on CPU
  start_timer(&t);
  blur_cpu(img, blur_img, width, height);
  stop_timer(&t);
  printf("CPU time: %f\n", time_diff(&t));
  write_rgb_image("__artifacts__/blur_image_cpu", blur_img, width, height);

  // on GPU
  start_timer(&t);
  blur_gpu(img, blur_img, width, height);
  cudaDeviceSynchronize();
  stop_timer(&t);
  printf("GPU time: %f\n", time_diff(&t));
  write_rgb_image("__artifacts__/blur_image_gpu", blur_img, width, height);

  // free memory
  free(img);
  free(blur_img);

  return 0;
}
