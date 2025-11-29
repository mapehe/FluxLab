#include "kernel/quantum/quantumKernels.cuh"
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

struct SquareMagnitude {
  __host__ __device__ float operator()(const cuFloatComplex &x) const {
    return cuCrealf(x) * cuCrealf(x) + cuCimagf(x) * cuCimagf(x);
  }
};

__global__ void scaleWavefunction(cuFloatComplex *d_psi, int totalElements,
                                  float scale) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < totalElements) {
    d_psi[idx].x *= scale;
    d_psi[idx].y *= scale;
  }
}

void normalizePsi(cuFloatComplex *d_psi, dim3 block, dim3 grid, int width,
                  int height, float dx, float dy) {
  int numElements = width * height;

  thrust::device_ptr<cuFloatComplex> th_psi(d_psi);
  float sumSq =
      thrust::transform_reduce(th_psi, th_psi + numElements, SquareMagnitude(),
                               0.0f, thrust::plus<float>());

  float currentProbability = sumSq * dx * dy;

  if (currentProbability == 0.0f)
    return; // Safety check
  float scaleFactor = 1.0f / sqrtf(currentProbability);

  scaleWavefunction<<<grid, block>>>(d_psi, numElements, scaleFactor);
  cudaDeviceSynchronize();
}

__global__ void initGaussian(cuFloatComplex *d_psi, int width, int height,
                             float dx, float dy, float x0, float y0,
                             float sigma, float kx, float ky, float amplitude) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int idx = y * width + x;

  if (x >= width || y >= height) {
    return;
  }

  const float center_x = width / 2.0f;
  const float center_y = height / 2.0f;
  const float scale = fminf(width, height) / 2.0f;

  const float nx = (x - center_x) / scale;
  const float ny = (center_y - y) / scale;

  float dist_sq = (nx - x0) * (nx - x0) + (ny - y0) * (ny - y0);
  float envelope = amplitude * expf(-dist_sq / (2.0f * sigma * sigma));

  float phase_angle = kx * nx + ky * ny;
  float cos_phase, sin_phase;
  sincosf(phase_angle, &sin_phase, &cos_phase);

  d_psi[idx] = make_cuFloatComplex(envelope * cos_phase, envelope * sin_phase);
}
