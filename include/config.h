#ifndef CONFIG_H
#define CONFIG_H

#include "json.hpp"

using json = nlohmann::json;

struct Params {
    int iterations;
    int gridWidth;
    int gridHeight;
    int threadsPerBlockX;
    int threadsPerBlockY;
    std::string outputFile;
};

Params preprocessParams(const json& j);

#endif
