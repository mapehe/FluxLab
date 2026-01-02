#include "engine/xyModelSimulation.cuh"
#include "io.h"
#include "kernel/langevin/langevin.cuh"
#include "kernel/random.cuh"
#include <cmath>

XYModelEngine::XYModelEngine(const Params &p)
    : ComputeEngine(p), d_grid(nullptr) {
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
}

XYModelEngine::~XYModelEngine() {
  if (d_grid) {
    cudaFree(d_grid);
    cudaFree(d_grid_tmp);
    cudaFree(d_states);
    cudaFree(d_neighbors);
    cudaFree(d_offsets);
    cudaFree(d_degrees);
    d_grid = nullptr;
  }
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
  const double T = params.xyModel.T * std::exp(-params.xyModel.tDecay * t);
  cudaMemcpy(d_grid_tmp, d_grid, bufferSize, cudaMemcpyDeviceToDevice);
  langevin_complex_update<<<grid, block>>>(
      d_grid_tmp, d_grid, d_neighbors, d_offsets, d_degrees, d_states, T,
      params.xyModel.dt, gridSize,
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
