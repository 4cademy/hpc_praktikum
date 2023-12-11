#include <stdlib.h>
#include <stdio.h>
#include "../functions.h"

int test_normal() {
    int n = 4;
    float* mat1 = (float*) malloc(n*n*sizeof(float));
    float* mat2 = (float*) malloc(n*n*sizeof(float));
    float* mat_out = malloc(n*n*sizeof(float));

    for (int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            mat1[i*n+j] = (float)(i+j);
            mat2[i*n+j] = (float)(i+j);
        }
    }

    float* mat2_golden = (float*) malloc(n*n*sizeof(float));

    mat2_golden[0] = 14;
    mat2_golden[1] = 20;
    mat2_golden[2] = 26;
    mat2_golden[3] = 32;
    mat2_golden[4] = 20;
    mat2_golden[5] = 30;
    mat2_golden[6] = 40;
    mat2_golden[7] = 50;
    mat2_golden[8] = 26;
    mat2_golden[9] = 40;
    mat2_golden[10] = 54;
    mat2_golden[11] = 68;
    mat2_golden[12] = 32;
    mat2_golden[13] = 50;
    mat2_golden[14] = 68;
    mat2_golden[15] = 86;

    normal_matrix_mul(mat1, mat2, mat_out, n);

    int is_equal = 1;
    for (int i = 0; i < n*n; i++) {
        if (mat2_golden[i] != mat_out[i]) {
            is_equal = 0;
            break;
        }
    }

    free(mat1);
    free(mat2);
    free(mat_out);
    free(mat2_golden);

    return is_equal;
}

int test_tiling() {
    int n = 8;
    int tiling_factor = 4;
    int tile_size = n/tiling_factor;
    float* mat1 = (float*) malloc(n*n*sizeof(float));
    float* mat2 = (float*) malloc(n*n*sizeof(float));
    float* mat_out = malloc(n*n*sizeof(float));
    float* mat2_golden = (float*) malloc(n*n*sizeof(float));

    for (int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            mat1[i*n+j] = (float)(i+j);
            mat2[i*n+j] = (float)(i+j);
            mat_out[i*n+j] = 0;
        }
    }

    normal_matrix_mul(mat1, mat2, mat2_golden, n);
    tiling_matrix_mul(mat1, mat2, mat_out, n, tiling_factor, tile_size);

    int is_equal = 1;
    for (int i = 0; i < n*n; i++) {
        if (mat2_golden[i] != mat_out[i]) {
            printf("i: %i, mat2_golden[i]: %f, mat_out[i]: %f\n", i, mat2_golden[i], mat_out[i]);
            is_equal = 0;
            break;
        }
    }

    free(mat1);
    free(mat2);
    free(mat_out);
    free(mat2_golden);

    return is_equal;
}


int main() {
    if (test_normal()) {
        printf("Normal matrix multiplication: PASSED\n");
        if (test_tiling()) {
            printf("Tiling matrix multiplication: PASSED\n");
        } else {
            printf("Tiling matrix multiplication: FAILED\n");
        }
    } else {
        printf("Normal matrix multiplication: FAILED\n");
        printf("No further tests will be executed\n");
    }
    return 0;
}