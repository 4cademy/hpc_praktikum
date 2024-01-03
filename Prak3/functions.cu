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
    int width = N;
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    for (int i = column; i < width; i += blockDim.x * gridDim.x) {
        for (int j = row; j < width; j += blockDim.y * gridDim.y) {
            float sum = 0;
            for (int k = 0; k < width; ++k) {
                sum += mat1[j * width + k] * mat2[k * width + i];
            }
            mat_out[j * width + i] = sum;
        }
    }

}

__global__ void matrixMul_test( float *mat_out, float *mat1, float *mat2, int N )
{
    int width = N;
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    for (int i = column; i < width; i += blockDim.x * gridDim.x) {
        for (int j = row; j < width; j += blockDim.y * gridDim.y) {
            float sum = 0;
            for (int k = 0; k < width; ++k) {
                sum += mat1[j * width + k] * mat2[k * width + i];
            }
            mat_out[j * width + i] = 2;
        }
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

    //TODO: delete print
    //printf("after giving to function\n Should be all 0\n");
    //for(int i = 0; i < n; i++) {
    //    for(int j = 0; j < n; j++) {
    //        printf("%f ", mat_out[i*n + j]);
    //    }
    //    printf("\n");
    //}


    int devId;
    cudaGetDevice(&devId);

    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devId);
    // cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devid);
    // matrixMul<<<32*numSMs, 256>>>(d_mat_out, d_mat1, d_mat2, n);
    matrixMul<<<32*numSMs, 1024>>>(d_mat_out, d_mat1, d_mat2, n);

    cudaMemcpy(mat_out, d_mat_out, size, cudaMemcpyDeviceToHost);

    //TODO: delete print statements
    // printf("mat1\n should have initialized matrix!\n");
    // for(int i = 0; i < n; i++) {
    //     for(int j = 0; j < n; j++) {
    //         printf("%f ", mat1[i*n + j]);
    //     }
    //     printf("\n");
    // }
    //
    // printf("mat2\n should have initialized matrix!\n");
    // for(int i = 0; i < n; i++) {
    //     for(int j = 0; j < n; j++) {
    //         printf("%f ", mat2[i*n + j]);
    //     }
    //     printf("\n");
    // }


    // printf("after giving back from gpu\n Should have result of computation!\n");
    // for(int i = 0; i < n; i++) {
    //     for(int j = 0; j < n; j++) {
    //         printf("%f ", mat_out[i*n + j]);
    //     }
    //     printf("\n");
    // }


    cudaFree(d_mat1);
    cudaFree(d_mat2);
    cudaFree(d_mat_out);
}