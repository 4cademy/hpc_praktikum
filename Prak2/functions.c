//
// Created by Marcel Beyer on 11.12.2023.
//

#include "functions.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

void mpi_mat_mul(const float* mat1, const float* mat2, int n, int start_mat_index, int end_mat_index, float* mat_out) {
    int start_i = floor(start_mat_index / n);
    int start_j = start_mat_index % n;
    int end_i = floor(end_mat_index / n);
    int end_j = end_mat_index % n;
    int out_index = 0;

    if (start_i == end_i) {
        // first row
        for(int j = start_j; j <= end_j; j++) {
            mat_out[out_index] = mat1[start_i*n] * mat2[j];
            for (int k = 1; k < n; k++) {
                mat_out[out_index] += mat1[start_i*n+k] * mat2[k*n+j];
            }
            out_index++;
        }
    } else {
        // first row
        for (int j = start_j; j < n; j++) {
            mat_out[out_index] = mat1[start_i * n] * mat2[j];
            for (int k = 1; k < n; k++) {
                mat_out[out_index] += mat1[start_i * n + k] * mat2[k * n + j];
            }
            out_index++;
        }

        for (int i = start_i + 1; i < end_i; i++) {
            for (int j = 0; j < n; j++) {
                mat_out[out_index] = mat1[i * n] * mat2[j];
                for (int k = 1; k < n; k++) {
                    mat_out[out_index] += mat1[i * n + k] * mat2[k * n + j];
                }
                out_index++;
            }
        }

        // last row
        for (int j = 0; j <= end_j; j++) {
            mat_out[out_index] = mat1[end_i * n] * mat2[j];
            for (int k = 1; k < n; k++) {
                mat_out[out_index] += mat1[end_i * n + k] * mat2[k * n + j];
            }
            out_index++;
        }
    }
}