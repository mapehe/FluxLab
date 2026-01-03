#include "engine/computeEngine.cuh"
#include "engine/xyModelSimulation.cuh"
#include "ml.cuh"
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <functional>

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

    return x;
  }
};

TORCH_MODULE(QNetwork);

void updatePolicy(
    QNetworkImpl *model, torch::optim::Optimizer &optimizer,
    const torch::Tensor &batch_states,  // Shape: [BatchSize, SimulationSteps, InputSize]
    const torch::Tensor &batch_actions, // Shape: [BatchSize, SimulationSteps] (Type: kLong)
    const torch::Tensor &batch_rewards  // Shape: [BatchSize, SimulationSteps]
) {
  model->train();

  auto input_size = batch_states.size(2);
  auto flat_states = batch_states.view({-1, input_size});
  auto logits = model->forward(flat_states);

  auto log_probs_all = torch::log_softmax(logits, /*dim=*/1);
  auto flat_actions = batch_actions.view({-1, 1});
  auto selected_log_probs = log_probs_all.gather(1, flat_actions);
  auto flat_rewards = batch_rewards.view({-1, 1});
  auto loss = -(selected_log_probs * flat_rewards).mean();

  optimizer.zero_grad();
  loss.backward();
  optimizer.step();
}

template <typename T, typename U> class ReinforcementLearningFramework {
protected:
  std::unique_ptr<ObservableComputeEngine<T, U>> simulator;
  QNetwork policy_net;
  torch::optim::Adam optimizer;
  torch::Device device;

  using SimulatorFactory =
      std::function<std::unique_ptr<ObservableComputeEngine<T, U>>(Params)>;
  SimulatorFactory makeSimulator;

public:
  ReinforcementLearningFramework(int state_dim, int action_dim,
                                 SimulatorFactory factory, Params params)
      : policy_net(state_dim, action_dim),
        optimizer(policy_net->parameters(), torch::optim::AdamOptions(1e-3)),
        device(torch::kCUDA), makeSimulator(factory) {
    policy_net->to(device);
    resetSimulator(params);
  }
  void step(int simulationStep, int batchIndex) {
    torch::NoGradGuard no_grad;
    auto observables = simulator->getObservable()->toVector();

    long input_size = static_cast<long>(observables.size());
    auto input = torch::from_blob(
            (void*)observables.data(), 
            {1, input_size}, 
            torch::kDouble
        );

    input = input.to(torch::kFloat).to(device);
    auto outoput = policy_net->forward(input);
    auto action = output.argmax(1).item<int>() - 1;

    simulator->modelAction(action);
    simulator->solveStep(simulationStep);
  }
  void setEval() { policy_net->eval(); }
  void resetSimulator(Params params) { simulator = makeSimulator(params); }
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

  constexpr int state_dim = 5;
  // Possible actions are {-1, 0, 1}
  const int action_dim = 3;

  auto factory = [=](Params config) { return std::make_unique<XYModelEngine>(config); };

  auto model =
      ReinforcementLearningFramework<cuDoubleComplex, XYModelObservable>(
          state_dim, action_dim, factory, config);

  for (int round = 0; round < config.xyModel.trainingRounds; round++) {
    std::cout << "Starting simulation round " << round + 1 << std::endl;

    torch::Tensor batch_states = torch::zeros({batch_size, config.xyModel.iterations, input_size});
    torch::Tensor batch_actions = torch::zeros({batch_size, config.xyModel.iterations, 1}, torch::kLong);
    torch::Tensor batch_rewards = torch::zeros({batch_size, config.xyModel.iterations, 1}, torch::kDouble);

    for (int batchIndex = 0; batchIndex < config.xyModel.trainingBatchSize;
         batchIndex++) {
      model.setEval();
      model.resetSimulator(config);
      for (int simulationStep = 0; simulationStep < config.xyModel.iterations; simulationStep++) {
        model.step(simulationStep, batchIndex);
      }
    }
  }
}
