#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

__global__ void simpleMatMul(float* A, float* B, float* C, int N) {
    // find thread idx
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= N || col >= N) {
        return;
    }

    float pvalue = 0;
    for (int k = 0; k < N; k++) {
        pvalue += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = pvalue;
}

// TODO: not implemented yet
__global__ void tiledMatMul(float* A, float* B, float* C, int N) {
}

void matMul(float* A, float* B, float* C, int N, int type) {

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, N * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, A, sizeof(float) * N * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, sizeof(float) * N * N, cudaMemcpyHostToDevice));

    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    switch (type) {
        case 1:
            simpleMatMul<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
            break;
        case 2:
            tiledMatMul<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
            break;
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

int main() {
    int N = 1024;

    // 1) initialize arrays
    float* A = new float[N * N];
    float* B = new float[N * N];
    float* C = new float[N * N];

    for (int i = 0; i < N * N; i++) {
        A[i] = i * 0.5f;
        B[i] = i * 1.5f;
    }

    // 2) feed them into the kernel function
    matMul(A, B, C, N, 1);

    delete[] A;
    delete[] B;
    delete[] C;
}
