#include <math.h>
#include <raylib.h>
#include <stdint.h>
#include <stdio.h>

#define NUM_BALLS 20

typedef struct {
  int32_t x;
  int32_t y;
  float dx;
  float dy;
  size_t r;
} ball_t;

__host__ __device__ size_t cdiv(size_t a, size_t b) { return (a + b - 1) / b; }

// shamelessly copied from raylib's source code HAHAHAHA
__device__ Color ColorFromHSV2(float hue, float saturation, float value) {
  Color color = {0, 0, 0, 255};

  // Red channel
  float k = fmodf((5.0f + hue / 60.0f), 6);
  float t = 4.0f - k;
  k = (t < k) ? t : k;
  k = (k < 1) ? k : 1;
  k = (k > 0) ? k : 0;
  color.r = (unsigned char)((value - value * saturation * k) * 255.0f);

  // Green channel
  k = fmodf((3.0f + hue / 60.0f), 6);
  t = 4.0f - k;
  k = (t < k) ? t : k;
  k = (k < 1) ? k : 1;
  k = (k > 0) ? k : 0;
  color.g = (unsigned char)((value - value * saturation * k) * 255.0f);

  // Blue channel
  k = fmodf((1.0f + hue / 60.0f), 6);
  t = 4.0f - k;
  k = (t < k) ? t : k;
  k = (k < 1) ? k : 1;
  k = (k > 0) ? k : 0;
  color.b = (unsigned char)((value - value * saturation * k) * 255.0f);

  return color;
}

__host__ __device__ inline float dist(int32_t x1, int32_t y1, int32_t x2,
                                      int32_t y2) {
  return sqrt((float)((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)));
}

float uniform(int32_t a, int32_t b) {
  return a + (float)rand() * (b - a) / RAND_MAX;
}

__global__ void metaball_kernel(Color *pixels, ball_t *balls,
                                size_t screenWidth, size_t screenHeight) {
  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  size_t y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= screenWidth || y >= screenHeight)
    return;

  float sum = 0;
  for (size_t i = 0; i < NUM_BALLS; ++i) {
    sum += balls[i].r / dist(balls[i].x, balls[i].y, x, y);
  }
  pixels[y * screenWidth + x] = ColorFromHSV2(sum * 360, 1.0f, 1.0f);
}

int main() {
  const int screenWidth = 1920;
  const int screenHeight = 1024;
  ball_t *balls;
  size_t ball_radius;

  ball_radius = 120;
  cudaMallocManaged(&balls, NUM_BALLS * sizeof(ball_t));

  // initialize balls
  for (size_t i = 0; i < NUM_BALLS; ++i) {
    balls[i] = {
        .x = (int32_t)uniform(0, screenWidth),
        .y = (int32_t)uniform(0, screenHeight),
        .dx = uniform(0, 15),
        .dy = uniform(0, 15),
        .r = ball_radius,
    };
  }

  Color *pixels;
  InitWindow(screenWidth, screenHeight, "Metaball");

  SetTargetFPS(24);

  cudaMallocManaged(&pixels, (screenWidth * screenHeight) * sizeof(Color));

  Image image = {0};
  image.width = screenWidth;
  image.height = screenHeight;
  image.format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8;
  image.mipmaps = 1;
  image.data = pixels;

  Texture2D texture = LoadTextureFromImage(image);

  while (!WindowShouldClose()) {

    dim3 numThreads(32, 32);
    dim3 numBlocks(cdiv(screenWidth, numThreads.x),
                   cdiv(screenHeight, numThreads.y));
    metaball_kernel<<<numBlocks, numThreads>>>(pixels, balls, screenWidth,
                                               screenHeight);

    for (size_t i = 0; i < NUM_BALLS; ++i) {
      balls[i].x += balls[i].dx;
      balls[i].y += balls[i].dy;
      if (balls[i].x < 0 || balls[i].x > screenWidth) {
        balls[i].dx = -balls[i].dx;
      }
      if (balls[i].y < 0 || balls[i].y > screenHeight) {
        balls[i].dy = -balls[i].dy;
      }
    }

    BeginDrawing();

    ClearBackground(RAYWHITE);

    UpdateTexture(texture, pixels);
    DrawTexture(texture, 0, 0, WHITE);

    EndDrawing();
  }

  CloseWindow();

  free(balls);
  cudaFree(balls);
  cudaFree(pixels);

  return 0;
}
