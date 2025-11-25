#!/bin/bash

set -e

nvcc hello.cu -o hello
./hello
