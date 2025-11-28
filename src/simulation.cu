#include "config.h"
#include "kernel/gpeKernel.h"
#include "kernel/testKernel.h"
#include "simulation.h"
#include <fstream>
#include <iostream>

std::unique_ptr<SimulationMode> getSimulationMode(const Params &params) {
  switch (params.kernelMode) {
  case CUDAKernelMode::Test:
    return std::make_unique<TestSimulation>(params);

  case CUDAKernelMode::GrossPitaevskii:
    return std::make_unique<GrossPitaevskiiSimulation>(params);

  default:
    throw std::runtime_error("Error: Invalid or unsupported CUDAKernelMode.");
  }
}

void run(json config) {
  std::cout << "[CPU] Preparing simulation..." << std::endl;
  const Params params = preprocessParams(config);
  auto sim = getSimulationMode(params);

  for (int t = 0; t < params.iterations; ++t) {
    sim->launch(t);
  }

  std::cout << "[CPU] Simulation complete." << std::endl;
  sim->saveResults(params.outputFile);
}
