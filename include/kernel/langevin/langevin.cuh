#ifndef LANGEVIN_KERNELS
#define LANGEVIN_KERNELS
#include "kernel/util.cuh"
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void langevin_complex_update(cuDoubleComplex *spins_in,
                                        cuDoubleComplex *spins_out,
                                        int *neighbors, int *offsets,
                                        int *degrees, curandState *states,
                                        double T, double dt, int num_spins,
                                        tmpGrid grid);

__global__ void calculate_energy_kernel(cuDoubleComplex *spins, int *neighbors,
                                        int *offsets, int *degrees,
                                        double *energy_out, int num_spins,
                                        tmpGrid grid);

__global__ void compute_vortex_density(cuDoubleComplex *spins,
                                       int *vortex_counts, int L, int N,
                                       tmpGrid grid);

#endif
