#include <mpi.h>
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include "functions.h"

#define NORMAL 0
#define LOOP_UNROLLING 1
#define LOOP_SWAP 2
#define TILING 3

const char * function_names[] = {
        "normal_matrix_mul",
        "loop_unrolling",
        "loop_swap",
        "tiling_matrix_mul"
};

int compare( const void* a, const void* b)
{
    long double int_a = * ( (long double*) a );
    long double int_b = * ( (long double*) b );

    if ( int_a == int_b ) return 0;
    else if ( int_a < int_b ) return -1;
    else return 1;
}


int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("Usage: %s <function number> <matrix size>\n", argv[0]);
        return 1;
    }
    if (atoi(argv[1]) < 0 || atoi(argv[1]) > 4){
        printf("Use a function through the defined numbers!\n");
        return 1;
    }
    if (atoi(argv[2]) < 0 || atoi(argv[2])%2 != 0){
        printf("Wrong Matrix size <n> needs to fulfill: 2^x = n !\n");
        return 1;
    }

    int function, matrix_size;
    //printf("argv[1]: %s\n", argv[1]);
    function = atoi(argv[1]);
    //printf("function: %i\n", function);
    matrix_size= atoi(argv[2]);

    int n = matrix_size;
    int evals = 10;
    struct timeval start, end;
    long double measures[10] = {0};
    long double avg_time = 0;

    float* mat1 = (float*) malloc(n*n*sizeof(float));
    float* mat2 = (float*) malloc(n*n*sizeof(float));
    int rank, size;
    MPI_Init(NULL,NULL);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    printf("rank macht matrix multiplikation: %i\n", rank);
    float* mat_out = malloc(n*n*sizeof(float));

    
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
            }
        }
        // actual matrix multiplication
        int tiling_factor = 4;
        int tile_size = n/tiling_factor;
        gettimeofday(&start, NULL);
        // tiling_matrix_mul(mat1, mat2, mat_out, n, tiling_factor, tile_size);
        // normal_matrix_mul(mat1, mat2, mat_out, n);
        // loop_unrolling(mat1, mat2, mat_out, n);
        // loop_swap(mat1, mat2_inv, mat_out, n);

        switch (function) {
            case NORMAL:
                normal_matrix_mul(mat1, mat2, mat_out, n);
                break;
            case LOOP_UNROLLING:
                loop_unrolling(mat1, mat2, mat_out, n);
                break;
            case LOOP_SWAP:
                loop_swap(mat1, mat2, mat_out, n);
                break;
            case TILING:
                tiling_matrix_mul(mat1, mat2, mat_out, n, tiling_factor, tile_size);
                break;
            default:
                printf("Invalid function!\n");
                return 1;
        }


        gettimeofday(&end, NULL);
        long long microseconds = ((end.tv_sec - start.tv_sec) * 1000*1000 + end.tv_usec - start.tv_usec);
        //printf("Time %i in us: %lld\n", e, microseconds);
        avg_time += (long double)(microseconds)/evals;

        long long tmp = (long long)n;
        long long no_fops = tmp * tmp * (2 * tmp - 1);
        measures[e] = (long double) no_fops / microseconds;

    }
    printf("Function: %s\n", function_names[function]);
    // printf("Avg. time: %Lf\n", avg_time);
    long long tmp = (long long)n;
    long long no_fops = tmp * tmp * (2 * tmp - 1);
    //printf("floating point operations: %lld\n", no_fops);
    //long double mega_flops = no_fops / avg_time;
    // printf("MFLOPS: %Lf\n\n", mega_flops);

    qsort( measures, 10, sizeof(long double), compare );
    printf("n x n Matrix mit n = %i\n", n);
    printf("Messungen: %i\n", evals);
    printf("in MFLOPS:\n");
    printf("Min: %Lf\n", measures[0]);
    printf("Median: %Lf\n", measures[5]);
    printf("Max: %Lf\n\n", measures[9]);

    free(mat1);
    free(mat2);
    free(mat_out);

    MPI_Finalize();
    return 0;
}