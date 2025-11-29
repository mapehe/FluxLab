#include "kernel/quantum/quantumKernels.cuh"

__global__ void initComplexPotential(cuComplex *d_V_tot, int width, int height,
                                     float dx, float dy, float trapFreqSq,
                                     float V_bias, float r_0, float sigma,
                                     float absorb_strength,
                                     float absorb_width) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int idx = y * width + x;

  if (x >= width || y >= height)
    return;

  float phys_x = (x - width / 2.0f) * dx;
  float phys_y = (height / 2.0f - y) * dy;
  float r = sqrtf(phys_x * phys_x + phys_y * phys_y);

  float v_harm = 0.5f * trapFreqSq * r * r;
  float v_waterfall = V_bias * tanhf((r - r_0) / sigma);
  float val_real = v_harm + v_waterfall + V_bias;

  float val_imag =
      -1.0f * absorb_strength * expf(-(r * r) / (absorb_width * absorb_width));

  d_V_tot[idx] = make_cuComplex(val_real, val_imag);
}
