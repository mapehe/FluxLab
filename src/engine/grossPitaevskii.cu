#include "engine/grossPitaevskii.h"
#include "io.h"

GrossPitaevskiiEngine::GrossPitaevskiiEngine(const Params &p)
    : ComputeEngine(p) {

  dt = p.dt;
  g = p.g;
  cufftPlan2d(&plan, height, width, CUFFT_C2C);

  size_t num_pixels = width * height;
  size_t size_bytes = num_pixels * sizeof(cuFloatComplex);

  cudaMalloc(&d_psi, size_bytes);
  cudaMemset(d_psi, 0, size_bytes);

  cudaMalloc(&d_V, size_bytes);
  cudaMemset(d_V, 0, size_bytes);

  cudaMalloc(&d_expK, size_bytes);
  cudaMemset(d_expK, 0, size_bytes);

  std::cout << "[Memory] Allocated Resources for grid " << p.gridWidth << "x"
            << p.gridHeight << ":\n"
            << " - Wavefunction (RW)\n"
            << " - Potential Grid (R)\n"
            << " - Kinetic Operator (R)\n"
            << std::endl;

  float dx = p.L / width;
  float dy = p.L / height;

  float L_x = width * dx;
  float L_y = height * dy;

  float dk_x = (2.0f * M_PI) / L_x;
  float dk_y = (2.0f * M_PI) / L_y;

  initGaussian<<<grid, block>>>(d_psi, width, height, dx, dy, p.x0, p.y0,
                                p.sigma, p.kx, p.ky, p.amp);
  cudaDeviceSynchronize();

  normalizePsi(d_psi, block, grid, width, height, dx, dy);
  cudaDeviceSynchronize();

  initComplexPotential<<<grid, block>>>(d_V, width, height, dx, dy, p.trapStr,
                                        p.V_bias, p.r_0, p.sigma2,
                                        p.absorbStrength, p.absorbWidth);
  cudaDeviceSynchronize();

  initKineticOperator<<<grid, block>>>(d_expK, width, height, dk_x, dk_y, dt);
  cudaDeviceSynchronize();
}

GrossPitaevskiiEngine::~GrossPitaevskiiEngine() {
  if (d_psi) {
    cudaFree(d_psi);
    d_psi = nullptr;
  }

  if (d_V) {
    cudaFree(d_V);
    d_V = nullptr;
  }

  if (d_expK) {
    cudaFree(d_expK);
    d_expK = nullptr;
  }

  cufftDestroy(plan);
}

void GrossPitaevskiiEngine::appendFrame(std::vector<cuFloatComplex> &history) {
  size_t frame_elements = width * height;
  size_t frame_bytes = frame_elements * sizeof(cuFloatComplex);

  size_t old_size = history.size();

  history.resize(old_size + frame_elements);

  cuFloatComplex *host_destination = history.data() + old_size;

  cudaMemcpy(host_destination, d_psi, frame_bytes, cudaMemcpyDeviceToHost);
}

void GrossPitaevskiiEngine::step(int t) {
  downloadIterator--;
  if (downloadIterator == 0) {
    appendFrame(h_data);
    downloadIterator = downloadFrequency;
  }

  float fft_scale = 1.0f / (float)(width * height);

  evolveRealSpace<<<grid, block>>>(d_psi, d_V, width, height, g, dt / 2.0f);
  cufftExecC2C(plan, d_psi, d_psi, CUFFT_FORWARD);
  evolveMomentumSpace<<<grid, block>>>(d_psi, d_expK, width, height, fft_scale);
  cufftExecC2C(plan, d_psi, d_psi, CUFFT_INVERSE);
  evolveRealSpace<<<grid, block>>>(d_psi, d_V, width, height, g, dt / 2.0f);
  cudaDeviceSynchronize();

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::stringstream ss;
    ss << "CUDA Error: " << cudaGetErrorString(err);
    throw std::runtime_error(ss.str());
  }
}

void GrossPitaevskiiEngine::saveResults(const std::string &filename) {
  saveToBinary(filename, this->h_data, this->width, this->height,
               this->iterations);
}
