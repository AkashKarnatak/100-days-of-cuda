#include <ATen/ATen.h>
#include <cstdint>
#include <stdint.h>

uint32_t cdiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

__global__ void mandelbrot_kernel(uint8_t *img, uint32_t h, uint32_t w,
                                  float xmin, float xmax, float ymin,
                                  float ymax) {
  uint32_t row = blockDim.y * blockIdx.y + threadIdx.y;
  uint32_t col = blockDim.x * blockIdx.x + threadIdx.x;
  if (row >= h || col >= w)
    return;

  // current pixel
  float cx = xmin + ((float)col / w) * (xmax - xmin);
  float cy = ymin + ((float)row / h) * (ymax - ymin);

  float zx = 0, zy = 0;
  float threshold = 4;
  uint32_t itr = 0, max_itr = 300;

  // z_n+1 = z_n ^ 2 + c
  while (zx * zx + zy * zy <= threshold && itr < max_itr) {
    float zx1 = zx * zx - zy * zy + cx;
    float zy1 = 2 * zx * zy + cy;
    zx = zx1, zy = zy1, ++itr;
  }

  img[row * w + col] = 255 * ((float)itr / max_itr);
}

at::Tensor mandelbrot(uint32_t h, uint32_t w, float xmin, float xmax,
                      float ymin, float ymax) {
  auto options = at::TensorOptions().dtype(at::kByte).device(at::kCUDA);
  at::Tensor img = at::empty({h, w}, options);

  dim3 numThreads(32, 32);
  dim3 numBlocks(cdiv(w, numThreads.x), cdiv(h, numThreads.y));
  mandelbrot_kernel<<<numBlocks, numThreads>>>(img.data_ptr<uint8_t>(), h, w,
                                               xmin, xmax, ymin, ymax);
  return img.cpu();
}
