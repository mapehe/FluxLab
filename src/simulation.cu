#include "config.h"
#include "engine/grossPitaevskii.cuh"
#include "engine/testSimulation.cuh"
#include "simulation.h"
#include <fstream>
#include <iostream>

void run(json config) {
  std::cout << "[CPU] Preparing simulation..." << std::endl;
  const Params params = preprocessParams(config);

  if (params.simulationMode == SimulationMode::GrossPitaevskii) {
    auto sim = std::make_unique<GrossPitaevskiiEngine>(params);
    sim->run();
    sim->saveResults(params.output);
    std::cout << "[CPU] Simulation complete." << std::endl;
    sim->saveResults(params.output);
  } else {
    auto sim = std::make_unique<TestEngine>(params);
    sim->run();
    sim->saveResults(params.output);
    std::cout << "[CPU] Simulation complete." << std::endl;
    sim->saveResults(params.output);
  }
}
