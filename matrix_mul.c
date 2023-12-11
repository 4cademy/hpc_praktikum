#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include "functions.h"

int main(int argc, char* argv[]) {
    int n = 512;
    printf("Time in [sec] for %ix%i:\n",n,n);
    int evals = 1;
    struct timeval start, end;
    long double avg_time = 0;

    float* mat1 = (float*) malloc(n*n*sizeof(float));
    float* mat2 = (float*) malloc(n*n*sizeof(float));
    float* mat2_inv = (float*) malloc(n*n*sizeof(float));
    float* mat_out = malloc(n*n*sizeof(float));

    
    for (int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            mat1[i*n+j] = (float)(i+j);
            mat2[i*n+j] = (float)(i+j);
            mat2_inv[j*n+i] = (float)(i+j);
        }
    }

    for (int e = 0; e < evals; e++) {
        // reset output matrix
        for (int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                mat_out[i*n+j] = 0;
            }
        }
        // actual matrix multiplication
        int tiling_factor = 4;
        int tile_size = n/tiling_factor;
        gettimeofday(&start, NULL);
        // tiling_matrix_mul(mat1, mat2, mat_out, n, tiling_factor, tile_size);
        normal_matrix_mul(mat1, mat2, mat_out, n);
        // loop_unrolling(mat1, mat2, mat_out, n);
        // loop_swap(mat1, mat2_inv, mat_out, n);

        gettimeofday(&end, NULL);
        long long microseconds = ((end.tv_sec - start.tv_sec) * 1000*1000 + end.tv_usec - start.tv_usec);
        printf("Time %i in us: %lld\n", e, microseconds);
        avg_time += (long double)(microseconds)/evals;
    }
    printf("Avg. time: %Lf\n", avg_time);
    long long tmp = (long long)n;
    long long no_fops = tmp * tmp * (2 * tmp - 1);
    printf("floating point operations: %lld\n", no_fops);
    long double mega_flops = no_fops / avg_time;
    printf("MFLOPS: %Lf\n", mega_flops);

    free(mat1);
    free(mat2);
    free(mat2_inv);
    free(mat_out);

    return 0;
}
