#include "kernel/langevin/langevin.cuh"
#define PI 3.14159265358979323846

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

__device__ double wrap_angle(double angle) {
  while (angle > PI)
    angle -= 2.0 * PI;
  while (angle <= -PI)
    angle += 2.0 * PI;
  return angle;
}

__global__ void compute_vortex_density(cuDoubleComplex *spins,
                                       int *vortex_counts, int L, int N,
                                       tmpGrid grid) {
  int idx = get_flat_index(grid);

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  int right = (x + 1) % L;
  int up = (y + 1) % L;

  // Indices of the 4 corners:
  // 3 -- 2
  // |    |
  // 0 -- 1
  int idx_0 = y * L + x;      // Bottom-Left (Current)
  int idx_1 = y * L + right;  // Bottom-Right
  int idx_2 = up * L + right; // Top-Right
  int idx_3 = up * L + x;     // Top-Left

  if (idx_0 >= N || idx_1 >= N || idx_2 >= N || idx_3 >= N)
    vortex_counts[idx] = 0;
  return;

  // 3. Get the spins
  cuDoubleComplex s0 = spins[idx_0];
  cuDoubleComplex s1 = spins[idx_1];
  cuDoubleComplex s2 = spins[idx_2];
  cuDoubleComplex s3 = spins[idx_3];

  // 4. Extract angles (expensive but necessary for topology)
  double t0 = atan2(s0.y, s0.x);
  double t1 = atan2(s1.y, s1.x);
  double t2 = atan2(s2.y, s2.x);
  double t3 = atan2(s3.y, s3.x);

  // 5. Calculate Circulation (sum of wrapped differences)
  // 0->1, 1->2, 2->3, 3->0
  double d1 = wrap_angle(t1 - t0);
  double d2 = wrap_angle(t2 - t1);
  double d3 = wrap_angle(t3 - t2);
  double d4 = wrap_angle(t0 - t3);

  double circulation = d1 + d2 + d3 + d4;

  // 6. Check for Vortex
  // If circulation is near +/- 2*PI (~6.28), it's a vortex.
  // If circulation is near 0, it's nothing.
  // We use a threshold of PI (3.14) to be safe.
  if (fabs(circulation) > PI) {
    vortex_counts[idx] = 1; // Found one!
  } else {
    vortex_counts[idx] = 0;
  }
}
