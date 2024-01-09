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
    int evals = 5;
    struct timeval start, end;
    long double* measures = (long double*) malloc(evals*sizeof(long double));
    long double avg_time = 0;


    float* mat2;
    float* mat_out;
    float* mat_golden;
    float* mat1;

    cudaError_t err;

    err = cudaMallocManaged((void**)&mat1, n*n*sizeof(float));
    if (err != cudaSuccess) {
        printf("Error allocating memory for mat1: %s\n", cudaGetErrorString(err));
        return 1;
    }
    err = cudaMallocManaged((void**)&mat2, n*n*sizeof(float));
    if (err != cudaSuccess) {
        printf("Error allocating memory for mat2: %s\n", cudaGetErrorString(err));
        return 1;
    }
    err = cudaMallocManaged((void**)&mat_out, n*n*sizeof(float));
    if (err != cudaSuccess) {
        printf("Error allocating memory for mat_out: %s\n", cudaGetErrorString(err));
        return 1;
    }
    err = cudaMallocManaged((void**)&mat_golden, n*n*sizeof(float));
    if (err != cudaSuccess) {
        printf("Error allocating memory for mat_golden: %s\n", cudaGetErrorString(err));
        return 1;
    };

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

        cudaMemPrefetchAsync(mat_out, n*n*sizeof(float), 0); // to GPU
        cudaMemPrefetchAsync(mat_golden, n*n*sizeof(float), 0); // to GPU

        //TODO: delete print
        printf("before giving to function\n Should be all 0\n");
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                printf("%f ", mat_out[i*n + j]);
            }
            printf("\n");
        }


        // actual matrix multiplication
        gettimeofday(&start, NULL);


        cuda_matrix_mul( mat1, mat2, mat_out, n);

        cudaMemPrefetchAsync(mat_out, n*n*sizeof(float), cudaCpuDeviceId); // back to CPU after computation


        gettimeofday(&end, NULL);
        long long microseconds = ((end.tv_sec - start.tv_sec) * 1000*1000 + end.tv_usec - start.tv_usec);
        avg_time += (long double)(microseconds)/evals;

        long long tmp = (long long)n;
        long long no_fops = tmp * tmp * (2 * tmp - 1);
        measures[e] = (long double) no_fops / microseconds;

    }
    qsort( measures, evals, sizeof(long double), compare );
    printf("n x n Matrix mit n = %i\n", n);
    printf("Messungen: %i\n", evals);
    printf("in MFLOPS:\n");
    printf("Min: %Lf\n", measures[0]);
    int median_index;
    long double median;
    if (evals % 2 == 0) {
        median_index = evals/2;
        median = measures[median_index];
    } else {
        median_index = (evals-1)/2;
        median = (measures[median_index] + measures[median_index+1])/2;
    }
    printf("Median: %Lf\n", median);
    printf("Max: %Lf\n\n", measures[evals-1]);



    cudaFree(mat1);
    cudaFree(mat2);
    cudaFree(mat_out);
    cudaFree(mat_golden);
    free(measures);

    return 0;
}