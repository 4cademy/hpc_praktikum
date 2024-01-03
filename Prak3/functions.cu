//
// Created by Marcel Beyer on 11.12.2023.
//

#include "functions.h"
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

void normal_matrix_mul(const float* mat1, const float* mat2, float* mat_out, int n) {
    for (int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            mat_out[i*n+j] = mat1[i*n] * mat2[j];
            for (int k = 1; k < n; k++) {
                mat_out[i*n+j] += mat1[i*n+k] * mat2[k*n+j];
            }
        }
    }
}

// void openmp_matrix_mul(const float* mat1, const float* mat2, float* mat_out, int n) {
// #pragma omp target data map(to:n,mat1[0:n*n],mat2[0:n*n]) map(from:mat_out[0:n*n])
//     {
// #pragma omp target teams distribute parallel for collapse(2)
//         for (int i = 0; i < n; i++) {
//             for(int j = 0; j < n; j++) {
//                 mat_out[i*n+j] = mat1[i*n] * mat2[j];
//                 for (int k = 1; k < n; k++) {
//                     mat_out[i*n+j] += mat1[i*n+k] * mat2[k*n+j];
//                 }
//             }
//         }
//     }
// }
__global__ void matrixMul( float *mat_out, float *mat1, float *mat2, int N )
{
//
//    _global__ void saxpy(int N, float a, float *x, float *y) { 
//        for (   int i = blockIdx.x * blockDim.x + threadIdx.x; // global thread ID (in x)
//                i < N; i += blockDim.x * gridDim.x ) { // stride of the loop
//                y[i] = a * x[i] + y[i];
//    }
//}
//
//    int row = blockIdx.y * blockDim.y + threadIdx.y;
//    int col = blockIdx.x * blockDim.x + threadIdx.x;
//
//    if( row < N && col < N )
//    {
//        float tmpSum = 0;
//
//        for (int i = 0; i < N; i++)
//        {
//            tmpSum += a[row * N + i] * b[i * N + col];
//        }
//
//        c[row * N + col] = tmpSum;
//    }
    for(int i = 0; i < N; i++){
        mat_out[i]= 2;
    }


}

void cuda_matrix_mul(const float* mat1, const float* mat2, float* mat_out, int n) {
    int size = n*n*sizeof(float);

    float* d_mat1;
    float* d_mat2;
    float* d_mat_out;

    cudaMalloc((void**)&d_mat1, size);
    cudaMalloc((void**)&d_mat2, size);
    cudaMalloc((void**)&d_mat_out, size);

    cudaMemcpy(d_mat1, mat1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat2, mat2, size, cudaMemcpyHostToDevice);

    //dim3 dimGrid(ceil(n/32.0), ceil(n/32.0), 1);
    //dim3 dimBlock(32, 32, 1);

    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 1);
    // cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devid);
    // matrixMul<<<32*numSMs, 256>>>(d_mat_out, d_mat1, d_mat2, n);
    matrixMul<<<1, 1>>>(d_mat_out, d_mat1, d_mat2, n);

    cudaMemcpy(mat_out, d_mat_out, size, cudaMemcpyDeviceToHost);

    cudaFree(d_mat1);
    cudaFree(d_mat2);
    cudaFree(d_mat_out);
}