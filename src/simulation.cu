#include "config.h"
#include "grid.h"
#include "simulation.h"
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>

__global__ void simulationKernel(cuFloatComplex *d_array, int gridWidth,
                                 int gridHeight, int time) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  if (x >= gridWidth || y >= gridHeight) {
    return;
  }

  const auto flat_index = get_flat_index(x, y, time, gridWidth, gridHeight);

  const float center_x = gridWidth / 2.0f;
  const float center_y = gridHeight / 2.0f;
  const float scale = fminf(gridWidth, gridHeight) / 2.0f;

  const float nx = (x - center_x) / scale;
  const float ny = (y - center_y) / scale;

  const float r = sqrtf(nx * nx + ny * ny);

  const float theta = atan2f(ny, nx);

  const float spatial_freq = 15.0f;
  const float temporal_freq = 0.05f;
  const float rotation_speed = 0.5f;

  const float phase =
      (r * spatial_freq) + (theta * rotation_speed) + (time * temporal_freq);

  const float real_part = cosf(phase);
  const float imag_part = sinf(phase);

  d_array[flat_index] = make_cuFloatComplex(real_part, imag_part);
}

void run(json config) {
  std::cout << "[CPU] Preparing simulation..." << std::endl;

  const Params params = preprocessParams(config);
  const auto d_array = allocateDeviceComplexArray(
      params.gridWidth, params.gridHeight, params.iterations);

  std::cout << "[CPU] Launching CUDA Kernel..." << std::endl;

  const dim3 threadsPerBlock(params.threadsPerBlockX, params.threadsPerBlockY);

  const dim3 numBlocks(
      (params.gridWidth + threadsPerBlock.x - 1) / threadsPerBlock.x,
      (params.gridHeight + threadsPerBlock.y - 1) / threadsPerBlock.y);

  const int progressStep =
      std::max(1, (int)std::round(params.iterations * 0.05));
  int nextReportIteration = progressStep;

  for (int t = 0; t < params.iterations; ++t) {
    if (t >= nextReportIteration) {
      double currentProgress = (double)t / params.iterations * 100.0;
      std::cout << "[CPU] Simulation Progress: "
                << std::round(currentProgress / 5.0) * 5.0
                << "% complete (Iteration " << t << ")\n";

      nextReportIteration += progressStep;
    }

    simulationKernel<<<numBlocks, threadsPerBlock>>>(d_array, params.gridWidth,
                                                     params.gridHeight, t);

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess) {
      std::stringstream ss;
      ss << "CUDA Error: " << cudaGetErrorString(err);
      throw std::runtime_error(ss.str());
    }

    cudaDeviceSynchronize();
  }

  std::cout << "[CPU] Simulation complete." << std::endl;

  std::vector<cuFloatComplex> h_array = getHostResults(
      d_array, params.gridWidth, params.gridHeight, params.iterations);

  cudaFree(d_array);

  saveToBinary(params.outputFile, h_array, params.gridWidth, params.gridHeight,
               params.iterations);
}
