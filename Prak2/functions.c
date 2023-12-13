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

void tiling_matrix_mul(const float* mat1, const float* mat2, float* mat_out, int n, int tiling_factor, int tile_size) {
    float* result_tile = malloc(tile_size*tile_size*sizeof(float));
    float* mat1_tile = malloc(tile_size*tile_size*sizeof(float));
    float* mat2_tile = malloc(tile_size*tile_size*sizeof(float));
    for (int tx_out = 0; tx_out < tiling_factor; tx_out++) {
        for (int ty_out = 0; ty_out < tiling_factor; ty_out++) {
            for (int tile_index = 0; tile_index < tiling_factor; tile_index++){
                // copy tile from mat1
                for (int i = 0; i < tile_size; i++) {
                    for (int j = 0; j < tile_size; j++) {
                        mat1_tile[i*tile_size+j] = mat1[(((ty_out*tile_size)+i)*n)+((tile_index * tile_size) + j)];
                    }
                }

                // copy tile from mat2
                for (int i = 0; i < tile_size; i++) {
                    for (int j = 0; j < tile_size; j++) {
                        mat2_tile[i*tile_size+j] = mat2[(((tile_index * tile_size) + i) * n) + ((tx_out * tile_size) + j)];
                    }
                }
                // multiply tiles
                normal_matrix_mul(mat1_tile, mat2_tile, result_tile, tile_size);
                // copy result tile to mat_out
                for (int i = 0; i < tile_size; i++) {
                    for (int j = 0; j < tile_size; j++) {
                        mat_out[(((ty_out*tile_size)+i)*n)+((tx_out * tile_size) + j)] += result_tile[i*tile_size+j];
                    }
                }
            }
        }
    }
    free(result_tile);
    free(mat1_tile);
    free(mat2_tile);
}

void loop_unrolling(const float* mat1, const float* mat2, float* mat_out, int n) {
    for (int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            for (int k = 0; k < n; k+=4) {
                mat_out[i*n+j] += mat1[i*n+k] * mat2[k*n+j];
                mat_out[i*n+j] += mat1[i*n+(k+1)] * mat2[(k+1)*n+j];
                mat_out[i*n+j] += mat1[i*n+(k+2)] * mat2[(k+2)*n+j];
                mat_out[i*n+j] += mat1[i*n+(k+3)] * mat2[(k+3)*n+j];
            }
        }
    }
}

void loop_swap(const float* mat1, const float* mat2, float* mat_out, int n) {
    for (int j = 0; j < n; j++) {
        for(int i = 0; i < n; i++) {
            mat_out[i*n+j] = mat1[i*n] * mat2[j];
            for (int k = 1; k < n; k++) {
                mat_out[i*n+j] += mat1[i*n+k] * mat2[k*n+j];
            }
        }
    }
}