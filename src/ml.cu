#include "ml.cuh"

void trainModel(Params config) {
  torch::Tensor tensor = torch::rand({2, 3});

  if (torch::cuda::is_available()) {
    std::cout << "[CPU] CUDA is available! Training on GPU." << std::endl;
    tensor = tensor.to(torch::kCUDA);
  } else {
    throw std::runtime_error("FATAL ERROR: CUDA is not available. Please check "
                             "your NVIDIA drivers and LibTorch version.");
  }
}
