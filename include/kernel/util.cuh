#ifndef UTIL_KERNELS
#define UTIL_KERNELS
#include <assert.h>

struct Grid {
    int width;
    int height;
};

__device__ __forceinline__ int get_flat_index(Grid args) {
    const auto [width, height] = args;

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  assert(i < width && j < height);

  return j * width + i;
}


struct Coords {
    float x;
    float y;
};

__device__ __forceinline__ Coords get_normalized_coords(Grid grid) {
    const auto [width, height] = grid;

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    const float center_x = width / 2.0f;
    const float center_y = height / 2.0f;
    const float scale    = fminf((float)width, (float)height) / 2.0f;

    return {
        .x = (i - center_x) / scale,
        .y = (center_y - j) / scale
    };
}

#endif
