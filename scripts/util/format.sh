#!/bin/bash

cd "$(dirname "$0")/../.."
find . -type f \( -name "*.cu" -o -name "*.h" \) -exec clang-format -i {} +
