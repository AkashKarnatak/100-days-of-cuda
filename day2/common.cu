#include "common.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

void create_rgb_image(const char *file_path, uint32_t width, uint32_t height) {
  struct pixel *img;

  img = (struct pixel *)malloc((width * height) * sizeof(struct pixel));
  assert(img != NULL);

  for (uint32_t i = 0; i < (width * height); ++i) {
    img[i].r = rand() % 255;
    img[i].g = rand() % 255;
    img[i].b = rand() % 255;
  }

  write_rgb_image(file_path, img, width, height);

  free(img);
}

void read_image(const char *file_path, struct pixel *img, uint32_t width,
                uint32_t height) {
  mkdir("./__artifacts__", 0755);
  if (access(file_path, F_OK) != 0) {
    create_rgb_image(file_path, width, height);
  }

  FILE *f;
  size_t ret;

  f = fopen(file_path, "r");
  assert(f != NULL);
  ret = fread(img, sizeof(struct pixel), (width * height), f);
  assert(ret == (width * height));
  fclose(f);
}

void write_image(const char *file_path, void *img, size_t size, uint32_t width,
                 uint32_t height) {
  FILE *f;
  size_t ret;

  f = fopen(file_path, "w");
  assert(f != NULL);
  ret = fwrite(img, size, width * height, f);
  assert(ret == (width * height));
  fclose(f);
}

void write_gray_image(const char *file_path, uint8_t *img, uint32_t width,
                      uint32_t height) {
  write_image(file_path, img, sizeof(uint8_t), width, height);
}

void write_rgb_image(const char *file_path, struct pixel *img, uint32_t width,
                     uint32_t height) {
  write_image(file_path, img, sizeof(struct pixel), width, height);
}

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
