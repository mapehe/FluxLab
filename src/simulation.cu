#include "config.h"
#include "sim/grossPitaevskii.h"
#include "sim/testSimulation.h"
#include "simulation.h"
#include <fstream>
#include <iostream>

std::unique_ptr<ComputeEngine> getComputeEngine(const Params &params) {
  switch (params.kernelMode) {
  case CUDAKernelMode::Test:
    return std::make_unique<TestEngine>(params);

  case CUDAKernelMode::GrossPitaevskii:
    return std::make_unique<GrossPitaevskiiEngine>(params);

  default:
    throw std::runtime_error("Error: Invalid or unsupported CUDAKernelMode.");
  }
}

void run(json config) {
  std::cout << "[CPU] Preparing simulation..." << std::endl;
  const Params params = preprocessParams(config);
  auto sim = getComputeEngine(params);

  for (int t = 0; t < params.iterations; ++t) {
    sim->step(t);
  }

  std::cout << "[CPU] Simulation complete." << std::endl;
  sim->saveResults(params.outputFile);
}
