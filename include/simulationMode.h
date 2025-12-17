#ifndef SIMULATION_MODE_H
#define SIMULATION_MODE_H

#include "json.hpp"

using json = nlohmann::json;

enum class SimulationMode { Test, GrossPitaevskii };

struct SimulationModeMap {
  static const std::unordered_map<std::string, SimulationMode> &get() {
    static const std::unordered_map<std::string, SimulationMode> map{
        {"test", SimulationMode::Test},
        {"grossPitaevskii", SimulationMode::GrossPitaevskii},
    };
    return map;
  }
};

inline void from_json(const nlohmann::json &j, SimulationMode &mode) {
  const std::string s = j.get<std::string>();
  const auto &map = SimulationModeMap::get();

  auto it = map.find(s);
  if (it != map.end()) {
    mode = it->second;
  } else {
    throw nlohmann::json::type_error::create(
        302, "Unknown SimulationMode: " + s, &j);
  }
}

#endif
