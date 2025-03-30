#ifndef COMMON_H
#define COMMON_H

#include <stdint.h>
#include <time.h>

struct timer {
  struct timespec start_time, end_time;
};

struct pixel {
  uint8_t r;
  uint8_t g;
  uint8_t b;
};

void create_rgb_image(const char *file_path, uint32_t width, uint32_t height);

void read_image(const char *file_path, struct pixel *img, uint32_t width,
                uint32_t height);

void write_gray_image(const char *file_path, uint8_t *img, uint32_t width,
                      uint32_t height);

void write_rgb_image(const char *file_path, struct pixel *img, uint32_t width,
                     uint32_t height);

void start_timer(struct timer *t);

void stop_timer(struct timer *t);

double time_diff(struct timer *t);

#endif
