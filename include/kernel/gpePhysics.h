#ifndef GPE_KERNEL_PHYSICS
#define GPE_KERNEL_PHYSICS

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

__global__ void initGaussian(cuFloatComplex *d_psi, int width, int height,
                             float dx, float dy, float x0, float y0,
                             float sigma, float kx, float ky, float amplitude);

void normalizePsi(cuFloatComplex *d_psi, dim3 block, dim3 grid, int width,
                  int height, float dx, float dy);

__global__ void initKineticOperator(cuFloatComplex *d_expK, int width,
                                    int height, float dk_x, float dk_y,
                                    float dt);

__global__ void evolveRealSpace(cuFloatComplex *d_psi, cuFloatComplex *d_V,
                                int width, int height, float g, float dt);

__global__ void evolveMomentumSpace(cuFloatComplex *d_psi,
                                    cuFloatComplex *d_expK, int width,
                                    int height, float scale);

__global__ void initComplexPotential(cuComplex *d_V_tot, int width, int height,
                                     float dx, float dy, float trapFreqSq,
                                     float V_bias, float r_0, float sigma,
                                     float absorb_strength, float absorb_width);

#endif
