#include "sim/grossPitaevskii.h"

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

__global__ void initKineticOperator(cuFloatComplex *d_expK, int width,
                                    int height, float dk_x, float dk_y,
                                    float dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = j * width + i;

  if (i >= width || j >= height)
    return;

  float kx;
  if (i <= width / 2) {
    kx = i * dk_x;
  } else {
    kx = (i - width) * dk_x;
  }

  float ky;
  if (j <= height / 2) {
    ky = j * dk_y;
  } else {
    ky = (j - height) * dk_y;
  }

  float k2 = kx * kx + ky * ky;
  float angle = -0.5f * k2 * dt;

  float c, s;
  sincosf(angle, &s, &c);
  d_expK[idx] = make_cuFloatComplex(c, s);
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

  printf("Initial Probability: %f. Scaling by: %f\n", currentProbability,
         scaleFactor);

  scaleWavefunction<<<grid, block>>>(d_psi, numElements, scaleFactor);
  cudaDeviceSynchronize();
}

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

__global__ void evolveRealSpace(cuFloatComplex *d_psi, cuFloatComplex *d_V,
                                int width, int height, float g, float dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = j * width + i;

  if (i >= width || j >= height)
    return;

  cuFloatComplex psi = d_psi[idx];
  cuFloatComplex V_c = d_V[idx];

  float V_real = cuCrealf(V_c);
  float V_imag = cuCimagf(V_c);

  float n = cuCrealf(psi) * cuCrealf(psi) + cuCimagf(psi) * cuCimagf(psi);

  float angle = -(V_real + g * n) * dt;
  float c, s;
  sincosf(angle, &s, &c);
  cuFloatComplex phasor = make_cuFloatComplex(c, s);

  float decay_factor = expf(V_imag * dt);

  cuFloatComplex psi_rotated = cuCmulf(psi, phasor);

  d_psi[idx] = make_cuFloatComplex(cuCrealf(psi_rotated) * decay_factor,
                                   cuCimagf(psi_rotated) * decay_factor);
}

__global__ void evolveMomentumSpace(cuFloatComplex *d_psi,
                                    cuFloatComplex *d_expK, int width,
                                    int height, float scale) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = j * width + i;

  if (i >= width || j >= height)
    return;

  cuFloatComplex psi = d_psi[idx];
  cuFloatComplex kOp = d_expK[idx];
  cuFloatComplex res = cuCmulf(psi, kOp);

  res.x *= scale;
  res.y *= scale;

  d_psi[idx] = res;
}
