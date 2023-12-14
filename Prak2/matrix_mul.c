#include <mpi.h>
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include "functions.h"
#include <math.h>

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
        printf("Usage: %s <function number> <matrix size>\n", argv[0]);
        return 1;
    }
    if (atoi(argv[1]) < 0 || atoi(argv[1])%2 != 0){
        printf("Wrong Matrix size <n> needs to fulfill: 2^x = n !\n");
        return 1;
    }

    int matrix_size;
    //printf("argv[1]: %s\n", argv[1]);
    matrix_size  = atoi(argv[1]);

    int n = matrix_size;
    int evals = 10;
    struct timeval start, end;
    long double measures[10] = {0};
    long double avg_time = 0;

    float* mat1 = (float*) malloc(n*n*sizeof(float));
    float* mat2 = (float*) malloc(n*n*sizeof(float));
    for (int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            mat1[i*n+j] = (float)(i+j);
            mat2[i*n+j] = (float)(i+j);
        }
    }

    int rank, size;



    // MPI section
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    for (int e = 0; e < evals; e++) {
        // start time measurement on rank 0
        if (rank == 0) {
            gettimeofday(&start, NULL);
        }

        int start_mat_index = floor(n * n / size * rank);
        int end_mat_index;
        if (rank == size - 1) {
            end_mat_index = n * n - 1;
        } else {
            end_mat_index = floor(n * n / size * (rank + 1)) - 1;
        }
        int sequence_length = end_mat_index - start_mat_index + 1;

        float *mat_out;
        if (rank == 0) {
            // printf("start time: %ld\n", start.tv_sec * 1000 * 1000 + start.tv_usec);
            mat_out = malloc(n * n * sizeof(float));
        } else {
            mat_out = malloc(sequence_length * sizeof(float));
        }

        mpi_mat_mul(mat1, mat2, n, start_mat_index, end_mat_index, mat_out);

        if (rank == 0) {
            // printf("rank %i hat fertig gerechnet\n", rank);
            for (int i = 1; i < size; i++) {
                int start_for_rank = floor(n * n / size * i);
                int end_for_rank;
                if (i == size - 1) {
                    end_for_rank = n * n - 1;
                } else {
                    end_for_rank = floor(n * n / size * (i + 1)) - 1;
                }
                int sequence_length_for_rank = end_for_rank - start_for_rank + 1;
                MPI_Recv(&mat_out[start_for_rank], sequence_length_for_rank, MPI_FLOAT, i, 0, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
            }
            gettimeofday(&end, NULL);
            // printf("end time: %ld\n", end.tv_sec * 1000 * 1000 + end.tv_usec);
            long long microseconds = ((end.tv_sec - start.tv_sec) * 1000 * 1000 + end.tv_usec - start.tv_usec);
            // printf("time: %lld\n", microseconds);
            avg_time += (long double)(microseconds)/evals;
            long long tmp = (long long)n;
            long long no_fops = tmp * tmp * (2 * tmp - 1);
            measures[e] = (long double) no_fops / microseconds;
        } else {
            MPI_Send(mat_out, sequence_length, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        }
        free(mat_out);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Finalize();

    if (rank == 0) {
        qsort( measures, 10, sizeof(long double), compare );
        printf("n x n Matrix mit n = %i\n", n);
        printf("Messungen: %i\n", evals);
        printf("in MFLOPS:\n");
        printf("Min: %Lf\n", measures[0]);
        printf("Median: %Lf\n", measures[5]);
        printf("Max: %Lf\n\n", measures[9]);
    }

    free(mat1);
    free(mat2);
    return 0;
}