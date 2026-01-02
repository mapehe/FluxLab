#ifndef ML_H
#define ML_H

#include "config.h"
#include "json.hpp"
#include <iostream>
#include <torch/torch.h>

using json = nlohmann::json;

void trainModel(Params config);

#endif
