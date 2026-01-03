#include "engine/xyModelSimulation.cuh"

const double TAX_VORTEX = 50.0;
const double TAX_MAG = 5.0;
const double TAX_TIME = 0.1;

double XYModelEngine::getStepLoss() {
  double cost_vortex = TAX_VORTEX * observable.vortexDensity;
  double cost_mag = TAX_MAG * (1.0 - observable.magnetizationMagnitude);
  double step_loss = cost_vortex + cost_mag + TAX_TIME;

  return step_loss;
}

void XYModelEngine::modelAction(int input) {
  observable.T +=
      ((double) input) * params.xyModel.thermostatSensitivity * params.xyModel.dt;
}

const XYModelObservable XYModelEngine::getObservable() { return observable; }

void XYModelEngine::computeObservables(tmpGrid gridParams) {
  calculate_energy_kernel<<<grid, block>>>(d_grid, d_neighbors, d_offsets,
                                           d_degrees, d_energy_out, gridSize,
                                           gridParams);
  cudaDeviceSynchronize();

  thrust::device_ptr<double> t_energy_ptr(d_energy_out);

  observable.totalEnergy = thrust::reduce(t_energy_ptr, t_energy_ptr + gridSize,
                                          0.0, thrust::plus<double>()) /
                           gridSize;

  thrust::device_ptr<cuDoubleComplex> dev_ptr(d_grid);
  cuDoubleComplex totalMagnetization =
      thrust::reduce(dev_ptr, dev_ptr + gridSize,
                     make_cuDoubleComplex(0.0, 0.0), ComplexSum());
  observable.magnetizationMagnitude = cuCabs(totalMagnetization) / gridSize;

  compute_vortex_density<<<grid, block>>>(
      d_grid, d_vortex_counts, params.xyModel.gridWidth, gridSize, gridParams);
  cudaDeviceSynchronize();

  thrust::device_ptr<int> ptr_vortex(d_vortex_counts);
  int total_vortices = thrust::reduce(ptr_vortex, ptr_vortex + gridSize);
  observable.vortexDensity = (double)total_vortices / (double)gridSize;
}

XYModelEngine::XYModelEngine(const Params &p)
    : ObservableComputeEngine(p), d_grid(nullptr) {
  bufferSize = params.xyModel.gridWidth * params.xyModel.gridHeight *
               sizeof(cuDoubleComplex);
  cudaError_t err = cudaMalloc(&d_grid, bufferSize);
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to allocate XYModelEngine device memory");
  }
  cudaMalloc(&d_grid_tmp, bufferSize);
  cudaMemset(d_grid, 0, bufferSize);

  grid = dim3(p.xyModel.threadsPerBlockX, p.xyModel.threadsPerBlockY);
  block = dim3((p.xyModel.gridWidth + grid.x - 1) / grid.x,
               (p.xyModel.gridHeight + grid.y - 1) / grid.y);

  auto gridWidth = p.xyModel.gridWidth;
  auto gridHeight = p.xyModel.gridHeight;
  gridSize = p.xyModel.gridWidth * p.xyModel.gridHeight;

  cudaMalloc(&d_energy_out, gridSize * sizeof(double));
  cudaMalloc((void **)&d_states, gridWidth * gridWidth * sizeof(curandState));

  init_rng_kernel<<<grid, block>>>(d_states, time(NULL), gridWidth);
  cudaDeviceSynchronize();

  init_complex_lattice_kernel<<<grid, block>>>(d_grid, d_states, gridWidth);
  cudaDeviceSynchronize();

  std::vector<int> h_neighbors;
  std::vector<int> h_offsets;
  std::vector<int> h_degrees;

  h_offsets.resize(gridSize);
  h_degrees.resize(gridSize);

  int edge_count = 0;

  for (int i = 0; i < gridSize; i++) {
    int x = i % gridWidth;
    int y = i / gridWidth;

    h_offsets[i] = edge_count;
    h_degrees[i] = 4;

    int right_x = (x + 1) % gridWidth;
    int left_x = (x - 1 + gridWidth) % gridWidth;

    int idx_right = (y * gridWidth) + right_x;
    int idx_left = (y * gridWidth) + left_x;

    int down_y = (y + 1) % gridHeight;
    int up_y = (y - 1 + gridHeight) % gridHeight;

    int idx_down = (down_y * gridWidth) + x;
    int idx_up = (up_y * gridWidth) + x;

    h_neighbors.push_back(idx_right);
    h_neighbors.push_back(idx_left);
    h_neighbors.push_back(idx_down);
    h_neighbors.push_back(idx_up);

    edge_count += 4;
  }

  cudaMalloc(&d_neighbors, h_neighbors.size() * sizeof(int));
  cudaMalloc(&d_offsets, gridSize * sizeof(int));
  cudaMalloc(&d_degrees, gridSize * sizeof(int));

  cudaMemcpy(d_neighbors, h_neighbors.data(), h_neighbors.size() * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_offsets, h_offsets.data(), gridSize * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_degrees, h_degrees.data(), gridSize * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMalloc(&d_vortex_counts, gridSize * sizeof(int));
  cudaMemset(d_vortex_counts, 0, gridSize * sizeof(int));

  computeObservables(
      {.width = params.xyModel.gridWidth, .height = params.xyModel.gridHeight});
  observable.simulationProgress = 0.0;
}

XYModelEngine::~XYModelEngine() {
  cudaFree(d_grid);
  cudaFree(d_grid_tmp);
  cudaFree(d_states);
  cudaFree(d_neighbors);
  cudaFree(d_offsets);
  cudaFree(d_degrees);
  cudaFree(d_energy_out);
  cudaFree(d_vortex_counts);
  d_grid = nullptr;
}

void XYModelEngine::appendFrame(std::vector<cuDoubleComplex> &history) {
  size_t frame_elements = params.xyModel.gridWidth * params.xyModel.gridHeight;
  size_t frame_bytes = frame_elements * sizeof(cuDoubleComplex);
  size_t old_size = history.size();

  history.resize(old_size + frame_elements);
  cuDoubleComplex *host_destination = history.data() + old_size;
  cudaMemcpy(host_destination, d_grid, frame_bytes, cudaMemcpyDeviceToHost);
}

void XYModelEngine::solveStep(int t) {
  observable.simulationProgress =
      ((double)t) / ((double)params.xyModel.iterations);
  if (!params.machineLearningMode) {
    observable.T = params.xyModel.T * std::exp(-params.xyModel.tDecay * t);
  }
  cudaMemcpy(d_grid_tmp, d_grid, bufferSize, cudaMemcpyDeviceToDevice);
  langevin_complex_update<<<grid, block>>>(
      d_grid_tmp, d_grid, d_neighbors, d_offsets, d_degrees, d_states,
      observable.T, params.xyModel.dt, gridSize,
      {.width = params.xyModel.gridWidth, .height = params.xyModel.gridHeight});
  computeObservables(
      {.width = params.xyModel.gridWidth, .height = params.xyModel.gridHeight});
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Kernel Launch Error: %s\n", cudaGetErrorString(err));
  }
  cudaDeviceSynchronize();
}

int XYModelEngine::getDownloadFrequency() {
  return params.xyModel.downloadFrequency;
}
int XYModelEngine::getTotalSteps() { return params.xyModel.iterations; }

void XYModelEngine::saveResults(const std::string &filename) {
  saveToBinaryJSON<cuDoubleComplex>(
      {.filename = filename,
       .data = historyData,
       .width = params.xyModel.gridWidth,
       .height = params.xyModel.gridHeight,
       .iterations = params.xyModel.iterations,
       .downloadFrequency = params.xyModel.downloadFrequency,
       .dtype = "complex128",
       .header = params.xyModel});
}
