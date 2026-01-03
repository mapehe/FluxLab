#include "engine/computeEngine.cuh"
#include "engine/xyModelSimulation.cuh"
#include "ml.cuh"
#include <cuComplex.h>
#include <cuda_runtime.h>

struct QNetworkImpl : torch::nn::Module {
  torch::nn::Linear fc1{nullptr}, fc2{nullptr}, out{nullptr};

  QNetworkImpl(int64_t input_size, int64_t action_dim) {
    fc1 = register_module("fc1", torch::nn::Linear(input_size, 64));
    fc2 = register_module("fc2", torch::nn::Linear(64, 64));
    out = register_module("out", torch::nn::Linear(64, action_dim));
  }

  torch::Tensor forward(torch::Tensor x) {
    x = torch::relu(fc1->forward(x));
    x = torch::relu(fc2->forward(x));
    x = out->forward(x);
    x = torch::tanh(x);

    return x;
  }
};

TORCH_MODULE(QNetwork);

template <typename T, typename U, typename I>
class ReinforcementLearningFramework {
protected:
  std::unique_ptr<ObservableComputeEngine<T, U, I>> simulator;
  QNetwork policy_net;
  torch::optim::Adam optimizer;
  torch::Device device;

public:
  ReinforcementLearningFramework(
      int state_dim, int action_dim,
      std::unique_ptr<ObservableComputeEngine<T, U, I>> ptr)
      : policy_net(state_dim, action_dim),
        optimizer(policy_net->parameters(), torch::optim::AdamOptions(1e-3)),
        device(torch::kCUDA), simulator(std::move(ptr)) {
    policy_net->to(device);
  }
};

void assertGPU() {
  if (torch::cuda::is_available()) {
    std::cout << "[CPU] CUDA is available! Training on GPU." << std::endl;
  } else {
    throw std::runtime_error("FATAL ERROR: CUDA is not available. Please check "
                             "your NVIDIA drivers and LibTorch version.");
  }
}

void assertXYModelMode(Params config) {
  if (config.simulationMode != SimulationMode::XYModel) {
    throw std::runtime_error("FATAL ERROR: Machine learning not implemented "
                             "for this simulation mode.");
  }
}

void trainModel(Params config) {
  assertGPU();
  assertXYModelMode(config);

  constexpr auto state_dim = 5;
  const int action_dim = 1;

  auto ptr = std::make_unique<XYModelEngine>(config);
  auto model = ReinforcementLearningFramework<cuDoubleComplex,
                                              XYModelObservable, double>(
      state_dim, action_dim, std::move(ptr));
}
