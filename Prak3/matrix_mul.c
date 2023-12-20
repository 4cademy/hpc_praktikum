#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include "functions.h"
#include <stdbool.h>

#define NORMAL 0
#define LOOP_UNROLLING 1
#define LOOP_SWAP 2
#define TILING 3

int compare( const void* a, const void* b)
{
    long double int_a = * ( (long double*) a );
    long double int_b = * ( (long double*) b );

    if ( int_a == int_b ) return 0;
    else if ( int_a < int_b ) return -1;
    else return 1;
}


int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s <matrix size>\n", argv[0]);
        return 1;
    }

    int matrix_size;
    matrix_size= atoi(argv[1]);

    int n = matrix_size;
    int evals = 1;
    struct timeval start, end;
    long double measures[10] = {0};
    long double avg_time = 0;

    float* mat1 = (float*) malloc(n*n*sizeof(float));
    float* mat2 = (float*) malloc(n*n*sizeof(float));
    float* mat_out = malloc(n*n*sizeof(float));
    float* mat_golden = malloc(n*n*sizeof(float));

    
    for (int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            mat1[i*n+j] = (float)(i+j);
            mat2[i*n+j] = (float)(i+j);
        }
    }

    for (int e = 0; e < evals; e++) {
        // reset output matrix
        for (int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                mat_out[i*n+j] = 0;
                mat_golden[i*n+j] = 0;
            }
        }
        // actual matrix multiplication
        int tiling_factor = 4;
        int tile_size = n/tiling_factor;
        gettimeofday(&start, NULL);

        openmp_matrix_mul(mat1, mat2, mat_out, n);

        gettimeofday(&end, NULL);
        long long microseconds = ((end.tv_sec - start.tv_sec) * 1000*1000 + end.tv_usec - start.tv_usec);
        avg_time += (long double)(microseconds)/evals;

        long long tmp = (long long)n;
        long long no_fops = tmp * tmp * (2 * tmp - 1);
        measures[e] = (long double) no_fops / microseconds;

    }
    qsort( measures, 10, sizeof(long double), compare );
    printf("n x n Matrix mit n = %i\n", n);
    printf("Messungen: %i\n", evals);
    printf("in MFLOPS:\n");
    printf("Min: %Lf\n", measures[0]);
    printf("Median: %Lf\n", measures[5]);
    printf("Max: %Lf\n\n", measures[9]);




    printf("\n");

    normal_matrix_mul(mat1, mat2, mat_out, n);
    bool is_equal = true;
    for (int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            if (mat_out[i*n+j] != mat_golden[i*n+j]) {
                is_equal = false;
                break;
            }
        }
    }
    if (is_equal) {
        printf("Matrix multiplication was correct\n");
    } else {
        printf("Matrix multiplication was incorrect\n");
    }

    free(mat1);
    free(mat2);
    free(mat_out);
    free(mat_golden);

    return 0;
}