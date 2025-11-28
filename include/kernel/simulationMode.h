#ifndef SIMULATION_MODE_H
#define SIMULATION_MODE_H

#include "config.h"

class SimulationMode {
public:
  virtual ~SimulationMode() = default;

  virtual void launch(int t) = 0;
  virtual void appendFrame(std::vector<cuFloatComplex> &history) = 0;
  virtual void saveResults(const std::string &filename) = 0;
};

#endif
