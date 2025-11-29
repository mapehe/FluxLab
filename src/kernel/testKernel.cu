#include "io.h"
#include "kernel/testKernel.cuh"
#include "kernel/util.cuh"

__host__ __device__ inline int get_flat_index(int x, int y, int gridWidth) {
  int W = gridWidth;
  return y * W + x;
}

__global__ void testKernel(cuFloatComplex *d_array, int gridWidth,
                           int gridHeight, int time) {
  int idx = get_flat_index({.width = gridWidth, .height = gridHeight});
  auto [nx, ny] =
      get_normalized_coords({.width = gridWidth, .height = gridHeight});

  const float r = sqrtf(nx * nx + ny * ny);

  const float theta = atan2f(ny, nx);

  const float spatial_freq = 15.0f;
  const float temporal_freq = 0.05f;
  const float rotation_speed = 0.5f;

  const float phase =
      (r * spatial_freq) + (theta * rotation_speed) + (time * temporal_freq);

  const float real_part = cosf(phase);
  const float imag_part = sinf(phase);

  d_array[idx] = make_cuFloatComplex(real_part, imag_part);
}
