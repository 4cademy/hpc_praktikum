//
// Created by Marcel Beyer on 11.12.2023.
//

#ifndef HPC_PRAKTIKUM_FUNCTIONS_H
#define HPC_PRAKTIKUM_FUNCTIONS_H

void normal_matrix_mul(const float* mat1, const float* mat2, float* mat_out, int n);

void tiling_matrix_mul(const float* mat1, const float* mat2, float* mat_out, int n, int tiling_factor, int tile_size);

void loop_unrolling(const float* mat1, const float* mat2, float* mat_out, int n);

void loop_swap(const float* mat1, const float* mat2, float* mat_out, int n);

#endif //HPC_PRAKTIKUM_FUNCTIONS_H
