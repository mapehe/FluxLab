#ifndef XY_MODEL_CUH
#define XY_MODEL_CUH

#include "config.h"
#include "engine/computeEngine.cuh"
#include "io.h"
#include "kernel/langevin/langevin.cuh"
#include "kernel/random.cuh"
#include "kernel/testKernel.cuh"
#include <cmath>
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

// TODO: Add T and others here
struct XYModelObservable {
    double T;
    double totalEnergy;
    double simulationProgress;
    cuDoubleComplex totalMagnetization;
    double vortexDensity;
};

class XYModelEngine : public ObservableComputeEngine<cuDoubleComplex, XYModelObservable> {
public:
  explicit XYModelEngine(const Params &p);
  ~XYModelEngine() override;
  void solveStep(int t) override;
  void appendFrame(std::vector<cuDoubleComplex> &history) override;
  void saveResults(const std::string &filename) override;
  int getDownloadFrequency() override;
  int getTotalSteps() override;

  // Need to access this from the ML model
  cuDoubleComplex *d_grid;

private:
  cuDoubleComplex *d_grid_tmp;
  curandState *d_states;
  dim3 grid;
  dim3 block;
  size_t bufferSize;
  int gridSize;

  int *d_neighbors, *d_offsets, *d_degrees;
  double *d_energy_out;

  void computeObservables(tmpGrid gridParams);
  int *d_vortex_counts;
};

#endif
