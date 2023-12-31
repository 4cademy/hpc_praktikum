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

void openmp_matrix_mul(const float* mat1, const float* mat2, float* mat_out, int n) {
#pragma omp target data map(to:n,mat1[0:n*n],mat2[0:n*n]) map(from:mat_out[0:n*n])
    {
#pragma omp target teams distribute parallel for collapse(2)
        for (int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                mat_out[i*n+j] = mat1[i*n] * mat2[j];
                for (int k = 1; k < n; k++) {
                    mat_out[i*n+j] += mat1[i*n+k] * mat2[k*n+j];
                }
            }
        }
    }
}