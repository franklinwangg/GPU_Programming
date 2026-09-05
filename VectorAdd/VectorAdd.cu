#include <cstdlib>;

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
    int N = 1 << 24;
    float* h_a = (float*)(std::malloc(N * sizeof(float)));
    float* h_b = (float*)(std::malloc(N * sizeof(float)));
    float* h_c = (float*)(std::malloc(N * sizeof(float)));


    
    for(int i = 0; i < N; i ++) {
        h_a[i] = i * 0.5f;
        h_b[i] = i * 1.5f;
    }

    stub_function(h_a, h_b, h_c, N);

    return 1;
}