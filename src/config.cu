#include "config.h"
#include <iostream>

template <typename T>
void get_and_validate_param(T &config_field, const json &j,
                            const std::string &key,
                            std::function<bool(const T &)> validator,
                            const std::string &validation_error_message) {
  try {
    config_field = j[key].get<T>();
  } catch (const nlohmann::json::type_error &e) {
    throw std::runtime_error(
        "Params Error: Field '" + key +
        "' is missing or has wrong type. Details: " + e.what());
  }

  if (!validator(config_field)) {
    throw std::runtime_error("Params Error: '" + key + "' " +
                             validation_error_message);
  }
}

Params preprocessParams(const json &j) {
  std::cout << "[Preprocess] Validating and calculating launch parameters..."
            << std::endl;

  Params config;

  const auto is_positive = [](const int &val) { return val > 0; };
  const char *const positive_number_message = "must be a positive number.";

  const auto is_not_empty = [](const std::string &val) { return !val.empty(); };
  const char *const not_empty_message = "cannot be empty.";

  get_and_validate_param<int>(config.iterations, j, "iterations", is_positive,
                              positive_number_message);

  get_and_validate_param<int>(config.gridWidth, j, "gridWidth", is_positive,
                              positive_number_message);

  get_and_validate_param<int>(config.gridHeight, j, "gridHeight", is_positive,
                              positive_number_message);

  get_and_validate_param<int>(config.threadsPerBlockX, j, "threadsPerBlockX",
                              is_positive, positive_number_message);

  get_and_validate_param<int>(config.threadsPerBlockY, j, "threadsPerBlockY",
                              is_positive, positive_number_message);

  get_and_validate_param<std::string>(config.outputFile, j, "outputFile",
                                      is_not_empty, not_empty_message);

  std::cout << "[Preprocess] Simulation configured to run for "
            << config.iterations << " iterations on a " << config.gridWidth
            << " x " << config.gridHeight << " grid." << std::endl;

  return config;
}
