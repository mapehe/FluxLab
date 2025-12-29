#include "kernel/quantum/quantumKernels.cuh"

__global__ void evolveRealSpace(cuDoubleComplex *d_psi, cuDoubleComplex *d_V,
                                int width, int height, float g, float dt) {
  int idx = get_flat_index({.width = width, .height = height});

  cuDoubleComplex psi = d_psi[idx];
  cuDoubleComplex V_c = d_V[idx];

  float V_real = V_c.x;
  float V_imag = V_c.y;

  float n = psi.x + psi.x + psi.y * psi.y;

  float angle = -(V_real + g * n) * dt;
  float c, s;
  sincosf(angle, &s, &c);
  cuDoubleComplex phasor = make_cuDoubleComplex(c, s);

  float decay_factor = expf(V_imag * dt);

  cuDoubleComplex psi_rotated = cuCmul(psi, phasor);

  d_psi[idx] = make_cuDoubleComplex(cuCreal(psi_rotated) * decay_factor,
                                    cuCimag(psi_rotated) * decay_factor);
}

__global__ void evolveMomentumSpace(cuDoubleComplex *d_psi,
                                    cuDoubleComplex *d_expK, int width,
                                    int height, float scale) {
  int idx = get_flat_index({.width = width, .height = height});

  cuDoubleComplex psi = d_psi[idx];
  cuDoubleComplex kOp = d_expK[idx];
  cuDoubleComplex res = cuCmul(psi, kOp);

  res.x *= scale;
  res.y *= scale;

  d_psi[idx] = res;
}

__global__ void initKineticOperator(cuDoubleComplex *d_expK,
                                    KineticInitArgs args) {
  const auto [width, height, dk_x, dk_y, dt] = args;
  int idx = get_flat_index({.width = width, .height = height});
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  float kx_val = (i <= width / 2) ? (float)i : (float)(i - width);
  float ky_val = (j <= height / 2) ? (float)j : (float)(j - height);

  float kx = kx_val * dk_x;
  float ky = ky_val * dk_y;

  float k2 = kx * kx + ky * ky;
  float angle = -0.5f * k2 * dt;

  float c, s;
  sincosf(angle, &s, &c);
  d_expK[idx] = make_cuDoubleComplex(c, s);
}
