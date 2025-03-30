#include "common.h"
#include <stdio.h>

const size_t BUFFER_SIZE = 100;

struct timer t;

void rgb2gray_cpu(struct pixel *img, uint8_t *gray_img, uint32_t width,
                  uint32_t height) {
  for (uint32_t i = 0; i < (width * height); ++i) {
    gray_img[i] = 0.21 * img[i].r + 0.72 * img[i].g + 0.07 * img[i].b;
  }
}

__global__ void rgb2gray_kernel(struct pixel *img, uint8_t *gray_img,
                                uint32_t width, uint32_t height) {
  uint32_t row = blockDim.y * blockIdx.y + threadIdx.y;
  uint32_t col = blockDim.x * blockIdx.x + threadIdx.x;
  if (row >= height || col >= width)
    return;
  uint32_t i = height * row + col;
  gray_img[i] = 0.21 * img[i].r + 0.72 * img[i].g + 0.07 * img[i].b;
}

void rgb2gray_gpu(struct pixel *img, uint8_t *gray_img, uint32_t width,
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
  rgb2gray_kernel<<<numBlocks, numThreads>>>(img, gray_img, width, height);

  // copy data from device to host
  cudaMemcpy(gray_img, gray_img_d, (width * height) * sizeof(uint8_t),
             cudaMemcpyDeviceToHost);

  // free memory
  cudaFree(img_d);
  cudaFree(gray_img_d);
}

int32_t main() {
  cudaDeviceSynchronize();

  char buf[BUFFER_SIZE];
  uint32_t width, height;
  struct pixel *img;
  uint8_t *gray_img;

  printf("=========== RGB2GRAY ===========\n");

  // read image
  width = 10240, height = 10240;
  img = (struct pixel *)malloc((width * height) * sizeof(struct pixel));
  gray_img = (uint8_t *)malloc((width * height) * sizeof(uint8_t));
  snprintf(buf, BUFFER_SIZE, "__artifacts__/rgb_image_%dx%d", width, height);
  read_image(buf, img, width, height);

  // on CPU
  start_timer(&t);
  rgb2gray_cpu(img, gray_img, width, height);
  stop_timer(&t);
  printf("CPU time: %f\n", time_diff(&t));
  write_gray_image("__artifacts__/gray_image_cpu", gray_img, width, height);

  // on GPU
  start_timer(&t);
  rgb2gray_gpu(img, gray_img, width, height);
  cudaDeviceSynchronize();
  stop_timer(&t);
  printf("GPU time: %f\n", time_diff(&t));
  write_gray_image("__artifacts__/gray_image_gpu", gray_img, width, height);

  // free memory
  free(img);
  free(gray_img);

  return 0;
}
