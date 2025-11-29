#include "engine/testSimulation.h"
#include "io.h"

TestEngine::TestEngine(const Params &p) : ComputeEngine(p), d_grid(nullptr) {
  size_t size = width * height * sizeof(cuFloatComplex);
  cudaError_t err = cudaMalloc(&d_grid, size);
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to allocate TestEngine device memory");
  }

  cudaMemset(d_grid, 0, size);

  std::cout << "[Helper] Allocated an array (" << width << "x" << height
            << ") on device." << std::endl;
}

TestEngine::~TestEngine() {
  if (d_grid) {
    cudaFree(d_grid);
    d_grid = nullptr;
  }
}

void TestEngine::appendFrame(std::vector<cuFloatComplex> &history) {
  size_t frame_elements = width * height;
  size_t frame_bytes = frame_elements * sizeof(cuFloatComplex);
  size_t old_size = history.size();

  history.resize(old_size + frame_elements);
  cuFloatComplex *host_destination = history.data() + old_size;
  cudaMemcpy(host_destination, d_grid, frame_bytes, cudaMemcpyDeviceToHost);
}

void TestEngine::step(int t) {
  downloadIterator--;
  if (downloadIterator == 0) {
    appendFrame(h_data);
    downloadIterator = downloadFrequency;
  }

  testKernel<<<grid, block>>>(d_grid, width, height, t);
  cudaDeviceSynchronize();

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::stringstream ss;
    ss << "CUDA Error in TestEngine: " << cudaGetErrorString(err);
    throw std::runtime_error(ss.str());
  }
}

void TestEngine::saveResults(const std::string &filename) {
  saveToBinary(filename, this->h_data, this->width, this->height,
               this->iterations);
}
