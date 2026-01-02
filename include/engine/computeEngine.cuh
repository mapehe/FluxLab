#ifndef COMPUTE_ENGINE_H
#define COMPUTE_ENGINE_H

#include "config.h"
#include <cuComplex.h>
#include <cuda_runtime.h>

template <typename T> class ComputeEngine {
protected:
  const Params params;
  int downloadIterator;

  std::vector<T> historyData;

  virtual int getDownloadFrequency() = 0;
  virtual int getTotalSteps() = 0;
  virtual void appendFrame(std::vector<T> &history) = 0;

  void step(int t) {
    if (downloadIterator == 0 && !params.machineLearningMode) {
      appendFrame(historyData);
    }
    downloadIterator = (downloadIterator + 1) % getDownloadFrequency();
    solveStep(t);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::stringstream ss;
      ss << "CUDA Error: " << cudaGetErrorString(err);
      throw std::runtime_error(ss.str());
    }
  }

public:
  ComputeEngine(const Params &p) : params(p), downloadIterator(0) {};
  virtual void solveStep(int t) = 0;
  virtual ~ComputeEngine() = default;
  virtual void saveResults(const std::string &filename) = 0;
  void run() {
    for (int t = 0; t < getTotalSteps(); t++) {
      step(t);
    }
  }
};

template <typename T, typename U>
class ObservableComputeEngine : public ComputeEngine<T> {
protected:
    U observable;

public:
    ObservableComputeEngine(const Params &p) : ComputeEngine<T>(p) {};
    virtual ~ObservableComputeEngine() = default;
};

#endif
