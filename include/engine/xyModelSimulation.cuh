#ifndef XY_MODEL_CUH
#define XY_MODEL_CUH

#include "config.h"
#include "engine/computeEngine.cuh"
#include "kernel/testKernel.cuh"
#include <curand_kernel.h>

class XYModelEngine : public ComputeEngine<cuDoubleComplex> {
public:
  explicit XYModelEngine(const Params &p);
  ~XYModelEngine() override;
  void solveStep(int t) override;
  void appendFrame(std::vector<cuDoubleComplex> &history) override;
  void saveResults(const std::string &filename) override;
  int getDownloadFrequency() override;
  int getTotalSteps() override;

private:
  cuDoubleComplex *d_grid;
  cuDoubleComplex *d_grid_tmp;
  curandState *d_states;
  dim3 grid;
  dim3 block;
  size_t bufferSize;
  int gridSize;

  int *d_neighbors, *d_offsets, *d_degrees;
};

#endif
