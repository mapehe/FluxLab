#ifndef GPE_KERNEL_CUH
#define GPE_KERNEL_CUH

#include "config.h"
#include "engine/computeEngine.h"
#include "kernel/quantum/quantumKernels.cuh"
#include <cufft.h>

class GrossPitaevskiiEngine : public ComputeEngine<cuFloatComplex> {
public:
  explicit GrossPitaevskiiEngine(const Params &p);
  ~GrossPitaevskiiEngine() override;
  void solveStep(int t) override;
  void appendFrame(std::vector<cuFloatComplex> &history) override;
  void saveResults(const std::string &filename) override;

private:
  cuFloatComplex *d_psi;
  cuFloatComplex *d_V;
  cuFloatComplex *d_expK;
  float dt;
  float g;
  cufftHandle plan;
};

#endif
