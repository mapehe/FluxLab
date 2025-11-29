#ifndef TEST_KERNEL_CUH
#define TEST_KERNEL_CUH

#include "config.h"
#include "engine/computeEngine.h"
#include "kernel/testKernel.h"

class TestEngine : public ComputeEngine<cuFloatComplex> {
public:
  explicit TestEngine(const Params &p);
  ~TestEngine() override;
  void step(int t) override;
  void appendFrame(std::vector<cuFloatComplex> &history) override;
  void saveResults(const std::string &filename) override;

private:
  cuFloatComplex *d_grid;
};

#endif
