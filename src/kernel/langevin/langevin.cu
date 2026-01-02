#include "kernel/langevin/langevin.cuh"

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

__global__ void calculate_energy_kernel(cuDoubleComplex *spins, int *neighbors,
                                        int *offsets, int *degrees,
                                        double *energy_out, int num_spins,
                                        tmpGrid grid) {
  int i = get_flat_index(grid);

  if (i >= num_spins)
    return;

  cuDoubleComplex my_spin = spins[i];

  double field_x = 0.0;
  double field_y = 0.0;

  int start = offsets[i];
  int count = degrees[i];

  for (int k = 0; k < count; k++) {
    int n_idx = neighbors[start + k];
    cuDoubleComplex n_spin = spins[n_idx];

    field_x += n_spin.x;
    field_y += n_spin.y;
  }

  double dot_product = (my_spin.x * field_x) + (my_spin.y * field_y);
  energy_out[i] = -0.5 * dot_product;
}
