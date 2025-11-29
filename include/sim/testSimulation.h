#ifndef TEST_KERNEL_CUH
#define TEST_KERNEL_CUH

#include "config.h"
#include "kernel/testPhysics.h"
#include "sim/simulationMode.h"

class TestEngine : public ComputeEngine {
public:
  explicit TestEngine(const Params &p);
  ~TestEngine() override;
  void step(int t) override;
  void appendFrame(std::vector<cuFloatComplex> &history) override;
  std::vector<cuFloatComplex> &getHistory() { return h_data; }
  void saveResults(const std::string &filename) override;

private:
  cuFloatComplex *d_grid;
  std::vector<cuFloatComplex> h_data;
};

#endif
