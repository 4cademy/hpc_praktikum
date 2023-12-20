//
// Created by Marcel Beyer on 11.12.2023.
//

#include "functions.h"
#include <stdlib.h>
#include <stdio.h>

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
    for (int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            mat_out[i*n+j] = mat1[i*n] * mat2[j];
            for (int k = 1; k < n; k++) {
                mat_out[i*n+j] += mat1[i*n+k] * mat2[k*n+j];
            }
        }
    }
}