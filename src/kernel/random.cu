#include "kernel/random.cuh"

__global__ void init_rng_kernel(curandState *states, unsigned long long seed,
                                int gridWidth) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = y * gridWidth + x;

  if (x < gridWidth && y < gridWidth) {
    curand_init(seed, idx, 0, &states[idx]);
  }
}

__global__ void init_complex_lattice_kernel(cuDoubleComplex *lattice,
                                            curandState *states,
                                            int gridWidth) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = y * gridWidth + x;

  if (x < gridWidth && y < gridWidth) {
    curandState localState = states[idx];
    float phi = curand_uniform(&localState) * 2.0f * M_PI;
    float c, s;
    __sincosf(phi, &s, &c);

    lattice[idx] = make_cuDoubleComplex(c, s);
    states[idx] = localState;
  }
}
