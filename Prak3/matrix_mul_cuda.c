#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include "functions.h"
#include <stdbool.h>

#ifdef __JETBRAINS_IDE__
#define __host__
#define __device__
#define __shared__
#define __constant__
#define __global__

// This is slightly mental, but gets it to properly index device function calls like __popc and whatever.
#define __CUDACC__
#include <device_functions.h>

// These headers are all implicitly present when you compile CUDA with clang. Clion doesn't know that, so
// we include them explicitly to make the indexer happy. Doing this when you actually build is, obviously,
// a terrible idea :D
#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_intrinsics.h>
#include <__clang_cuda_math_forward_declares.h>
#include <__clang_cuda_complex_builtins.h>
#include <__clang_cuda_cmath.h>
#endif // __JETBRAINS_IDE__

__global__ void matrixMul( int *c, int *a, int *b, int N )
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if( row < N && col < N )
    {
        int tmpSum = 0;

        for (int i = 0; i < N; i++)
        {
            tmpSum += a[row * N + i] * b[i * N + col];
        }

        c[row * N + col] = tmpSum;
    }
}

int main( int argc, char* argv[] )
{
    if (argc != 2) {
        printf("Usage: %s <matrix size>\n", argv[0]);
        return 1;
    }

    int matrix_size;
    int evals = 1;

    matrix_size= atoi(argv[1]);

    int n = matrix_size;

    int size = n*n*sizeof(float);


    float* mat1 = (float*) malloc(size);
    float* mat2 = (float*) malloc(size);
    float* mat_out = malloc(size);
    float* mat_golden = malloc(size);

    for (int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            mat1[i*n+j] = (float)(i+j);
            mat2[i*n+j] = (float)(i+j);
        }
    }

    for (int e = 0; e < evals; e++) {
        // reset output matrix
        for (int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                mat_out[i*n+j] = 0;
                mat_golden[i*n+j] = 0;
            }
        }

    cudaError_t err;

    // Allocate memory on the device
    float *a, *b, *c;

    cudaMalloc((void**) &a, size);
    cudaMalloc((void**) &b, size);
    cudaMalloc((void**) &c, size);

    //gpu pointers
    // Copy host memory to device
    cudaMemcpy(a,mat1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(b,mat2, size, cudaMemcpyHostToDevice);

    // Perform matrix multiplication on the device
    dim3 dimGrid(n, n);
    dim3 dimBlock(1, 1);

    matrixMul<<<dimGrid, dimBlock>>>(mat3, mat1, mat2, n);

    // Copy results from device to host
    cudaMemcpy( mat_out,c, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(mat1);
    cudaFree(mat2);
    cudaFree(mat3);

    return 0;
}