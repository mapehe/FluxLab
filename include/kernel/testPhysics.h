#ifndef TEST_PHYSICS
#define TEST_PHYSICS

__global__ void testKernel(cuFloatComplex *d_array, int gridWidth,
                           int gridHeight, int time);

#endif
