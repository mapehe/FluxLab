#ifndef CONFIG_H
#define CONFIG_H

#include "json.hpp"
#include "simulationMode.h"
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>

using json = nlohmann::json;

struct TestParams {
  int iterations;
  int gridWidth;
  int gridHeight;
  int threadsPerBlockX;
  int threadsPerBlockY;
  int downloadFrequency;

  NLOHMANN_DEFINE_TYPE_INTRUSIVE(TestParams, iterations, gridWidth, gridHeight,
                                 threadsPerBlockY, threadsPerBlockY,
                                 downloadFrequency)
};

struct GrossPitaevskiiParams {
  int iterations;
  int gridWidth;
  int gridHeight;
  int threadsPerBlockX;
  int threadsPerBlockY;
  int downloadFrequency;
  float L;
  float sigma;
  float x0;
  float y0;
  float kx;
  float ky;
  float amp;
  float omega;
  float trapStr;

  float dt;
  float g;

  float V_bias;
  float r_0;
  float sigma2;
  float absorbStrength;
  float absorbWidth;

  NLOHMANN_DEFINE_TYPE_INTRUSIVE(GrossPitaevskiiParams, iterations, gridWidth,
                                 gridHeight, threadsPerBlockY, threadsPerBlockY,
                                 downloadFrequency, L, sigma, x0, y0, kx, ky,
                                 amp, omega, trapStr, dt, g, V_bias, r_0,
                                 sigma2, absorbStrength, absorbWidth)
};

struct XYModelParams {
  int iterations;
  int gridWidth;
  int gridHeight;
  int threadsPerBlockX;
  int threadsPerBlockY;
  int downloadFrequency;
  float dt;
  float T;

  NLOHMANN_DEFINE_TYPE_INTRUSIVE(XYModelParams, iterations, gridWidth,
                                 gridHeight, threadsPerBlockY, threadsPerBlockY,
                                 dt, T, downloadFrequency)
};

struct Params {
  std::string output;
  SimulationMode simulationMode;

  TestParams test;
  GrossPitaevskiiParams grossPitaevskii;
  XYModelParams xyModel;
};

Params preprocessParams(const json &j);

#endif
