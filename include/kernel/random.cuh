#ifndef RANDOM_CUH
#define RANDOM_CUH

#include <cuComplex.h>
#include <curand_kernel.h>

__global__ void init_rng_kernel(curandState *states, unsigned long long seed,
                                int gridWidth);

__global__ void init_complex_lattice_kernel(cuDoubleComplex *lattice,
                                            curandState *states, int gridWidth);

#endif
