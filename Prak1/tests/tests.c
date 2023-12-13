#include <stdlib.h>
#include <stdio.h>
#include "../functions.h"

void print_differences(float* mat1, float* mat2, int n) {
    printf("mat2_golden:\n");
    for (int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            printf("%d\t", (int)(mat1[i*n+j]));
        }
        printf("\n");
    }
    printf("mat_out:\n");
    for (int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            if (mat1[i*n+j] != mat2[i*n+j]) {
                printf("%d\t", (int)(mat2[i*n+j]));
            } else {
                printf("EQL\t");
            }
        }
        printf("\n");
    }
}

int check_equal(float* mat1, float* mat2, int n) {
    int is_equal = 1;
    for (int i = 0; i < n*n; i++) {
        if (mat1[i] != mat2[i]) {
            is_equal = 0;
            break;
        }
    }

    if (!is_equal) {
        print_differences(mat1, mat2, n);
    }
    return is_equal;
}

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

    int is_equal = check_equal(mat2_golden, mat_out, n);

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

    int is_equal = check_equal(mat2_golden, mat_out, n);

    free(mat1);
    free(mat2);
    free(mat_out);
    free(mat2_golden);

    return is_equal;
}

int test_loop_unrolling(){
    int n = 8;
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
    loop_unrolling(mat1, mat2, mat_out, n);

    int is_equal = check_equal(mat2_golden, mat_out, n);

    free(mat1);
    free(mat2);
    free(mat_out);
    free(mat2_golden);

    return is_equal;
}

int test_loop_swap(){
    int n = 8;
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
    loop_swap(mat1, mat2, mat_out, n);

    int is_equal = check_equal(mat2_golden, mat_out, n);


    free(mat1);
    free(mat2);
    free(mat_out);
    free(mat2_golden);

    return is_equal;
}

int main() {
    if (test_normal()) {
        printf("Normal: PASSED\n");
        if (test_tiling()) {
            printf("Tiling: PASSED\n");
        } else {
            printf("Tiling: FAILED\n");
        }
        if (test_loop_unrolling()) {
            printf("Loop unrolling: PASSED\n");
        } else {
            printf("Loop unrolling: FAILED\n");
        }
        if (test_loop_swap()) {
            printf("Loop swap: PASSED\n");
        } else {
            printf("Loop swap: FAILED\n");
        }
    } else {
        printf("Normal: FAILED\n");
        printf("No further tests will be executed\n");
    }
    return 0;
}