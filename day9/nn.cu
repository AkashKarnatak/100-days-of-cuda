#include <raylib.h>
#include <cstdlib>
#include <cstring>
#include <stdint.h>
#include <stdlib.h>

#define IMG_SIZE 28
#define RECT_SIZE 30

uint8_t img[IMG_SIZE * IMG_SIZE];

void display(uint8_t *img) {
  InitWindow(IMG_SIZE * RECT_SIZE, IMG_SIZE * RECT_SIZE, "MNIST");

  SetTargetFPS(60);

  while (!WindowShouldClose()) {
    BeginDrawing();
    ClearBackground(BLACK);
    for (uint32_t i = 0; i < IMG_SIZE; ++i) {
      for (uint32_t j = 0; j < IMG_SIZE; ++j) {
        Color c = {img[i * IMG_SIZE + j], img[i * IMG_SIZE + j],
                   img[i * IMG_SIZE + j], 255};
        DrawRectangle(i * RECT_SIZE, j * RECT_SIZE, RECT_SIZE, RECT_SIZE,
                      c);
      }
    }
    EndDrawing();
  }

  CloseWindow();
}

#define TILE_SIZE 32

__host__ __device__ uint32_t cdiv(uint32_t a, uint32_t b) {
  return (a + b - 1) / b;
}

typedef struct {
  float *data;
  size_t w;
  size_t h;
  size_t n;
} Mat;

typedef struct {
  float *data;
  size_t n;
} Vec;

__global__ void matmul_kernel(Mat *A, Mat *B, Mat *C) {
  size_t row = blockDim.y * blockIdx.y + threadIdx.y;
  size_t col = blockDim.x * blockIdx.x + threadIdx.x;

  __shared__ float A_s[TILE_SIZE][TILE_SIZE];
  __shared__ float B_s[TILE_SIZE][TILE_SIZE];

  float sum = 0.0f;
  for (size_t tileIdx = 0; tileIdx < cdiv(A->w, TILE_SIZE); ++tileIdx) {
    if (row < A->h && tileIdx * TILE_SIZE + threadIdx.x < A->w)
      A_s[threadIdx.y][threadIdx.x] =
          A->data[row * A->w + tileIdx * TILE_SIZE + threadIdx.x];
    else
      A_s[threadIdx.y][threadIdx.x] = 0;
    if (tileIdx * TILE_SIZE + threadIdx.y < B->h && col < B->w)
      B_s[threadIdx.y][threadIdx.x] =
          B->data[(tileIdx * TILE_SIZE + threadIdx.y) * B->w + col];
    else
      B_s[threadIdx.y][threadIdx.x] = 0;

    __syncthreads();

    for (size_t i = 0; i < TILE_SIZE; ++i) {
      sum += A_s[threadIdx.y][i] * B_s[i][threadIdx.x];
    }

    __syncthreads();
  }
  if (row < C->h && col < C->w)
    C->data[row * C->w + col] = sum;
}

__global__ void matadd_kernel(Mat *A, Mat *B, Mat *C) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= A->n)
    return;

  C->data[i] = A->data[i] + B->data[i];
}

__global__ void sigmoid_kernel(Mat *A, Mat *B) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= A->n)
    return;

  B->data[i] = 1.0f / (1.0f + expf(-A->data[i]));
}

__global__ void transpose_kernel(Mat *A, Mat *B) {
  size_t row = blockDim.y * blockIdx.y + threadIdx.y;
  size_t col = blockDim.x * blockIdx.x + threadIdx.x;

  B->data[col * A->h + row] = A->data[row * A->w + col];
}

void mat_free(Mat *X) { free(X->data); }

Mat matmul(Mat *A, Mat *B) {
  Mat C;
  return C;
}

Vec matmul(Mat *A, Vec *u) {
  Vec v;
  return v;
}

Mat matadd(Mat *A, Mat *B) {
  Mat C;
  return C;
}

Vec vecadd(Vec *u, Vec *v) {
  Vec w;
  return w;
}

Mat mat_sigmoid(Mat *A) {
  Mat C;
  return C;
}

Mat mat_transpose(Mat *A) {
  Mat C;
  return C;
}

void xavier_init(Mat *X, size_t height, size_t width) {
  X->h = height, X->w = width;
  X->n = height * width;
  X->data = (float *)malloc(X->n * sizeof(float));

  for (size_t row = 0; row < X->h; ++row) {
    for (size_t col = 0; col < X->w; ++col) {
      // TODO: fix xavier initialzation
      X->data[row * X->w + col] = (float)rand() / RAND_MAX;
    }
  }
}

void zero_init(Mat *X, size_t height, size_t width) {
  X->h = height, X->w = width;
  X->n = height * width;
  X->data = (float *)calloc(X->n, sizeof(float));
}

Mat Linear(Mat *X, size_t out_dims) {
  // X -> B x N
  size_t B = X->h, N = X->w, M = out_dims;

  Mat weight, bias;

  // initialize
  xavier_init(&weight, N, M); // N x M
  zero_init(&bias, N, M);     // N x M

  Mat Y = matmul(X, &weight); // B x M
  Mat Z = matadd(&Y, &bias);  // B x M

  mat_free(&weight);
  mat_free(&bias);

  return Z;
}

Vec Linear(Vec *u, size_t out_dims) {
  size_t N = u->n, M = out_dims;

  Mat weight, bias;

  // initialize
  xavier_init(&weight, N, M); // N x M
  zero_init(&bias, N, M);     // N x M

  Vec v = matmul(&weight, u); // B x M
  Vec w = vecadd(u, &v);      // B x M

  mat_free(&weight);
  mat_free(&bias);

  return w;
}

void inference(Vec img) { Vec out = Linear(&img, 512); }

void nn(Mat imgs) {
  Mat out = Linear(&imgs, 512);
  Mat logits = Linear(&out, 10);
}

int main() {
  for (uint32_t i = 0; i < IMG_SIZE * IMG_SIZE; ++i) {
    img[i] = rand() % 255;
  }

  display(img);

  return 0;
}
