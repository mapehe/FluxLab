#ifndef LINALG_H
#define LINALG_H

#include <cuComplex.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float distanceSq(float2 p1, float2 p2) {
  float dx = p1.x - p2.x;
  float dy = p1.y - p2.y;
  return fmaf(dx, dx, dy * dy);
}

#endif
