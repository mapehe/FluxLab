#ifndef GRID_H
#define GRID_H

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <stdexcept>
#include <iostream>
#include <string>
#include <fstream>
#include "json.hpp"
using json = nlohmann::json;

__host__ __device__ int get_flat_index(int x, int y, int t, int gridWidth, int gridHeight);

cuFloatComplex* allocateDeviceComplexArray(size_t x, size_t y, size_t t) {
    size_t total_elements = x * y * t;
    size_t mem_size = total_elements * sizeof(cuFloatComplex); 
    cuFloatComplex* d_array = nullptr;

    cudaError_t cudaStatus = cudaMalloc((void**)&d_array, mem_size);
    cudaMemset(d_array, 0, mem_size);

    if (cudaStatus != cudaSuccess) {
        std::string error_msg = "Allocation failed for ";
        
        std::cerr << "ERROR: cudaMalloc failed inside allocateDeviceComplexArray: " 
                  << cudaGetErrorString(cudaStatus) << std::endl;
        error_msg += std::to_string(x) + "x" + std::to_string(y) + "x" + std::to_string(t) + " array): ";
        error_msg += cudaGetErrorString(cudaStatus);
        throw std::runtime_error(error_msg);
    }

    std::cout << "[Helper] Allocated an array (" << x << "x" << y << "x" << t
      << ") on device." << std::endl;
              
    return d_array;
}

__host__ __device__ int get_flat_index(int x, int y, int t, int gridWidth, int gridHeight) {
    int W = gridWidth;
    int H = gridHeight;
    int slice_size = W * H; 
    
    return t * slice_size + y * W + x;
}

std::vector<cuFloatComplex> getHostResults(cuFloatComplex* d_array, size_t x, size_t y, size_t t) {
    size_t total_elements = x * y * t;
    size_t mem_size = total_elements * sizeof(cuFloatComplex);
    
    std::vector<cuFloatComplex> h_array(total_elements);

    cudaError_t cudaStatus = cudaMemcpy(h_array.data(), d_array, mem_size, cudaMemcpyDeviceToHost);
    
    if (cudaStatus != cudaSuccess) {
        std::string error_msg = "cudaMemcpy failed to copy results to host: ";
        error_msg += cudaGetErrorString(cudaStatus);
        throw std::runtime_error(error_msg);
    }
    
    std::cout << "[Helper] Copied device array to host memory." << std::endl;
    return h_array;
}

void saveToBinary(const std::string& filename, const std::vector<cuFloatComplex>& data, 
                  int width, int height, int iterations) {
    
    std::ofstream out(filename, std::ios::out | std::ios::binary);
    if (!out) {
        throw std::runtime_error("Could not open file for writing");
    }

    std::cout << "[Helper] Writing binary output..." << std::endl;

    out.write(reinterpret_cast<const char*>(&width), sizeof(int));
    out.write(reinterpret_cast<const char*>(&height), sizeof(int));
    out.write(reinterpret_cast<const char*>(&iterations), sizeof(int));

    size_t dataSize = data.size() * sizeof(cuFloatComplex);
    out.write(reinterpret_cast<const char*>(data.data()), dataSize);

    out.close();
    std::cout << "[Helper] Saved " << (dataSize / 1024.0 / 1024.0) << " MB to " << filename << std::endl;
}


#endif
