#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

// Macro for GPU error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// CUDA Kernel: Vector Addition (C = A + B)
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    // Array size (1 million elements)
    const int N = 1000000;
    const size_t bytes = N * sizeof(float);

    // 1. Allocate Host (CPU) memory
    std::vector<float> h_A(N, 1.0f); // Filled with 1.0
    std::vector<float> h_B(N, 2.0f); // Filled with 2.0
    std::vector<float> h_C(N, 0.0f); // Destination array

    // 2. Allocate Device (GPU) memory
    float* d_A = nullptr, * d_B = nullptr, * d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    // 3. Copy data from Host to Device
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    // 4. Configure Grid and Block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    std::cout << "Launching kernel with " << blocksPerGrid
        << " blocks and " << threadsPerBlock << " threads per block...\n";

    // 5. Launch Kernel
    vectorAdd << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 6. Copy result from Device back to Host
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    // 7. Verify the output
    bool correct = true;
    for (int i = 0; i < N; ++i) {
        if (std::abs(h_C[i] - 3.0f) > 1e-5) {
            correct = false;
            std::cout << "Error at index " << i << ": expected 3.0, got " << h_C[i] << "\n";
            break;
        }
    }

    if (correct) {
        std::cout << "SUCCESS! All " << N << " vector additions calculated correctly on the GPU.\n";
    }

    // 8. Free Device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}