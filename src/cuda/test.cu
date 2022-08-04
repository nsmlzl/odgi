#include <stdio.h>
#include <cuda/cudaflow.hpp>

__global__ void cuda_hello_device() {
    printf("Hello World from CUDA device\n");
}

void cuda_hello_host() {
    printf("Hello World from CUDA host\n");
    cuda_hello_device<<<1,5>>>();
    cudaDeviceSynchronize();
    return;
}
