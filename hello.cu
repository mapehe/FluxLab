#include <iostream>
#include <cuda_runtime.h>

__global__ void helloKernel(char *msg) {
    msg[threadIdx.x] = "Hello, World!\n"[threadIdx.x];
}

int main() {
    const char hostMsg[] = "Hello, World!\n";
    const int msgSize = sizeof(hostMsg);

    char *devMsg;
    cudaMalloc((void**)&devMsg, msgSize);

    // Launch kernel with one block and msgSize threads
    helloKernel<<<1, msgSize>>>(devMsg);

    char result[msgSize];
    cudaMemcpy(result, devMsg, msgSize, cudaMemcpyDeviceToHost);

    std::cout << result;

    cudaFree(devMsg);
    return 0;
}
