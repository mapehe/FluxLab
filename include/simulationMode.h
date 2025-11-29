#ifndef SIMULATION_MODE_H
#define SIMULATION_MODE_H

#include "json.hpp"

using json = nlohmann::json;

enum class SimulationMode { Test, GrossPitaevskii };

inline void from_json(const nlohmann::json &j, SimulationMode &mode) {
  static const std::unordered_map<std::string, SimulationMode> str_to_enum{
      {"test", SimulationMode::Test},
      {"grossPitaevskii", SimulationMode::GrossPitaevskii},
  };

  const std::string s = j.get<std::string>();
  auto it = str_to_enum.find(s);

  if (it != str_to_enum.end()) {
    mode = it->second;
  } else {
    throw nlohmann::json::type_error::create(
        302, "Unknown SimulationMode: " + s, &j);
  }
}

#endif
