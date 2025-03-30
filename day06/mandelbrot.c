#include <stdint.h>

#define MAX_WIDTH 3840
#define MAX_HEIGHT 2160

typedef struct {
  uint8_t r;
  uint8_t g;
  uint8_t b;
  uint8_t a;
} Color;

uint8_t img_buf[4 * MAX_WIDTH * MAX_HEIGHT];

const uint32_t max_itr = 100;

Color palette[17] = {
    {66, 30, 15, 255},    // Dark brown
    {25, 7, 26, 255},     // Dark purple
    {9, 1, 47, 255},      // Deeper purple
    {4, 4, 73, 255},      // Even deeper blue
    {0, 7, 100, 255},     // Darker blue
    {12, 44, 138, 255},   // Medium blue
    {24, 82, 177, 255},   // Brighter blue
    {57, 125, 209, 255},  // Light blue
    {134, 181, 229, 255}, // Very light blue
    {211, 236, 248, 255}, // Almost white blue
    {241, 233, 191, 255}, // Light yellow
    {248, 201, 95, 255},  // Yellow
    {255, 170, 0, 255},   // Orange
    {204, 128, 0, 255},   // Dark orange
    {153, 87, 0, 255},    // Darker orange
    {106, 52, 3, 255},    // Darkest orange
    {0, 0, 0, 255}        // Black
};

uint8_t *get_img_buf() { return img_buf; }

uint32_t check_mandelbrot(float cx, float cy) {
  float zx = 0, zy = 0;
  uint32_t itr = 0;
  while (zx * zx + zy * zy <= 4 && itr < max_itr) {
    float z2x = zx * zx - zy * zy;
    float z2y = 2 * zx * zy;
    zx = z2x + cx, zy = z2y + cy, ++itr;
  }
  return itr;
}

void generate_mandelbrot(uint32_t width, uint32_t height, float x0, float y0,
                         float scale_factor) {
  for (uint32_t row = 0; row < height; ++row) {
    for (uint32_t col = 0; col < width; ++col) {
      float cx = x0 + (col - width / 2.0) * scale_factor;
      float cy = y0 + (row - height / 2.0) * scale_factor;
      uint32_t itr = check_mandelbrot(cx, cy);
      Color c = itr == max_itr ? palette[16] : palette[itr % 16];
      img_buf[4 * (row * width + col)] = c.r;
      img_buf[4 * (row * width + col) + 1] = c.g;
      img_buf[4 * (row * width + col) + 2] = c.b;
      img_buf[4 * (row * width + col) + 3] = c.a;
    }
  }
}
