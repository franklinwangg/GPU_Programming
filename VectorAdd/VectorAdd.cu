#include <cstdlib>;
#include <chrono>;
#include <stdio.h>;

#define CUDA_CHECK(call) {\
    cudaError_t error = call;\
    if(error != cudaSuccess) {\
        printf(cudaGetErrorString());\
        exit(1);\
    }\
}\


__global__ void vector_add(float* d_a, float* d_b, float* d_c, int N) {
    // 1) find thread
    int index = threadIdx.x + blockDim.x * blockIdx.x;

    // 2) perform calculation
    if(index < N) {
        d_c[index] = d_a[index] + d_b[index];
    }
}

void stub_function(float* h_a, float* h_b, float* h_c, int N) {
    // 1) allocate memory
    float* d_a;
    //  = (float*)(std::malloc(N * sizeof(float)));
    float* d_b;
    //  = (float*)(std::malloc(N * sizeof(float)));
    float* d_c;
    //  = (float*)(std::malloc(N * sizeof(float)));

    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));

    // float* h_c = (float*)(std::malloc(N * sizeof(float)));

    // 2) copy over to GPU
    cudaMemcpy(d_a, h_a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(float) * N, cudaMemcpyHostToDevice);
    
    // 3) do kernel
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    vector_add<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    // 4) copy back
    cudaMemcpy(h_c, d_c, sizeof(float) * N, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

int main() {
    int exponents[] = {24, 28};
    int numTests = sizeof(exponents) / sizeof(exponents[0]);

    for(int t = 0; t < numTests; t++) {
        int e = exponents[t];
        int N = 1 << e;

        float* h_a = (float*)(std::malloc(N * sizeof(float)));
        float* h_b = (float*)(std::malloc(N * sizeof(float)));
        float* h_c = (float*)(std::malloc(N * sizeof(float)));

        for(int i = 0; i < N; i ++) {
            h_a[i] = i * 0.5f;
            h_b[i] = i * 1.5f;
        }

        // compare GPU performance against CPU
        // 1) CPU time
        auto start = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < N; i ++) {
            h_c[i] = h_a[i] + h_b[i];
        }
        auto end = (std::chrono::high_resolution_clock::now());

        std::chrono::duration<double, std::milli> elapsed = end - start;

        // 2) GPU time
        cudaEvent_t GPUstart;
        cudaEvent_t GPUstop;
        float elapsedTime;
        cudaEventCreate(&GPUstart);
        cudaEventCreate(&GPUstop);
        cudaEventRecord(GPUstart);
        stub_function(h_a, h_b, h_c, N);
        cudaEventRecord(GPUstop);

        cudaEventSynchronize(GPUstop);
        cudaEventElapsedTime(&elapsedTime, GPUstart, GPUstop);
        cudaEventDestroy(GPUstart);
        cudaEventDestroy(GPUstop);

        printf("N = 1<<%d (%d elements): CPU = %f ms, GPU = %f ms\n", e, N, elapsed.count(), elapsedTime);

        std::free(h_a);
        std::free(h_b);
        std::free(h_c);
    }

    return 0;
}