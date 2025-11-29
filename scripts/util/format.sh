#!/bin/bash

cd "$(dirname "$0")/../.."
find . -type f \( -name "*.cu " -o -name "*.cpp" -o -name "*.cuh" -o -name "*.h" -o -name "*.json" \) -exec clang-format -i {} +
