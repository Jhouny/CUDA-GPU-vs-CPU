#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include<chrono>
#include <iostream>

using namespace std::chrono;

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
void addWithCPU(int* c, const int* a, const int* b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
    const int arraySize = 1<<20;
    int* A = (int*)malloc(arraySize * sizeof(int));
    int* B = (int*)malloc(arraySize * sizeof(int));
    int* C = (int*)malloc(arraySize * sizeof(int));

    for (auto i = 0; i < arraySize; i++) {
        A[i] = i;       // 0, 1, 2, 3, ...
        B[i] = 10 * i;  // 0, 10, 20, 30, ...
    }

    // Add vectors in parallel and measure time.
    auto t1 = high_resolution_clock::now();
    cudaError_t cudaStatus = addWithCuda(C, A, B, arraySize);
    auto t2 = high_resolution_clock::now();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }
    
    duration<double, std::milli> ms = t2 - t1;  // Calculates dT in ms as double
    std::cout << "CUDA: {";
    for (auto i = 0; i < 5; i++)
        std::cout << A[i] << ",";
    std::cout << "} + {";
    for (auto i = 0; i < 5; i++)
        std::cout << B[i] << ",";
    std::cout << "} = {";
    for (auto i = 0; i < 5; i++)
        std::cout << C[i] << ",";
    std::cout << "}\t\t DT: " << ms.count() << std::endl;


    for (auto i = 0; i < 5; i++) {
        C[i] = 0;
    }

    // Add vectors in series and measure time.
    t1 = high_resolution_clock::now();
    addWithCPU(C, A, B, arraySize);
    t2 = high_resolution_clock::now();

    ms = t2 - t1;  // Calculates dT in ms as double
    std::cout << "CPU: {";
    for (auto i = 0; i < 5; i++)
        std::cout << A[i] << ",";
    std::cout << "} + {";
    for (auto i = 0; i < 5; i++)
        std::cout << B[i] << ",";
    std::cout << "} = {";
    for (auto i = 0; i < 5; i++)
        std::cout << C[i] << ",";
    std::cout << "}\t\t DT: " << ms.count() << std::endl;

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    int blockSize;    // The launch configurator returned block size
    int minGridSize;  // The minimum grid size needed to achieve the maximum occupancy for a full device launch
    int gridSize;     // The actual grid size needed, based on input size 

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, addKernel, 0, size); 
    // Round up according to array size
    gridSize = (size + blockSize - 1) / blockSize; 

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output).
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<gridSize, blockSize>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy result failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}


// Function using CPU to add vector serially
void addWithCPU(int* c, const int* a, const int* b, unsigned int size) {
    for (auto i = 0; i < size; i++)
        c[i] = a[i] + b[i];
}
