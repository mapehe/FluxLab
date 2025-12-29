#ifndef IO_H
#define IO_H

#include "json.hpp"
#include <cuComplex.h>
#include <fstream>
#include <stdexcept>
#include <vector>

using json = nlohmann::json;

template <typename ComplexT> struct SaveOptions {
  std::string filename;
  const std::vector<ComplexT> &data;
  int width;
  int height;
  int iterations;
  int downloadFrequency;
  std::string dtype;
  json header;
};

template <typename ComplexT>
inline void saveToBinaryJSON(const SaveOptions<ComplexT> &opts) {
  const auto &[filename, data, width, height, iterations, downloadFrequency,
               dtype, header] = opts;

  std::ofstream out(filename, std::ios::out | std::ios::binary);
  if (!out)
    throw std::runtime_error("Could not open file");

  json tmp = header;
  json version = json({{"commit", COMMIT_HASH}, {"dtype", dtype}});
  tmp.merge_patch(version);
  std::string headerStr = tmp.dump();

  out.write(headerStr.c_str(), headerStr.size());
  out.write("\n", 1);

  out.write(reinterpret_cast<const char *>(data.data()),
            data.size() * sizeof(ComplexT));

  out.close();
}

#endif
