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
                                        tmpGrid grid) {
  int i = get_flat_index(grid);

  if (i >= num_spins)
    return;

  cuDoubleComplex my_spin = spins_in[i];
  cuDoubleComplex force_field = make_cuDoubleComplex(0.0, 0.0);

  int start = offsets[i];
  int count = degrees[i];

  for (int k = 0; k < count; k++) {
    int n_idx = neighbors[start + k];
    cuDoubleComplex n_spin = spins_in[n_idx];

    force_field.x += n_spin.x;
    force_field.y += n_spin.y;
  }

  double torque = (my_spin.x * force_field.y) - (my_spin.y * force_field.x);

  curandState localState = states[i];
  double noise = curand_normal_double(&localState);
  states[i] = localState;

  double d_theta = (torque * dt) + (sqrt(2.0 * T * dt) * noise);

  cuDoubleComplex new_spin;
  new_spin.x = my_spin.x - (my_spin.y * d_theta);
  new_spin.y = my_spin.y + (my_spin.x * d_theta);

  double inv_mag =
      1.0 / sqrt(new_spin.x * new_spin.x + new_spin.y * new_spin.y);
  new_spin.x *= inv_mag;
  new_spin.y *= inv_mag;

  spins_out[i] = new_spin;
}
#endif
