#include <stdio.h>
#include <time.h>
#include <stdlib.h>

int main() {
    int n = 1024;
    printf("Time in [sec] for %ix%i:\n",n,n);
    int evals = 10;
    clock_t start, stop;
    double time_taken = 0;
    double total_time = 0;

    float* mat1 = (float*) malloc(n*n*sizeof(float));
    float* mat2 = (float*) malloc(n*n*sizeof(float));
    float* mat_out = malloc(n*n*sizeof(float));

    
    for (int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            mat1[i*n+j] = rand();
            mat2[i*n+j] = rand();
        }
    }

    for (int e = 0; e < evals; e++) {
        // actual matrix multiplication
        start = clock();
        for (int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                mat_out[i*n+j] = mat1[i*n] * mat2[j];
                for (int k = 1; k < n; k++) {
                    mat_out[i*n+j] += mat1[i*n+k] * mat2[k*n+j];
                }
            }
        }
        stop = clock();
        time_taken = (double)(stop - start) / CLOCKS_PER_SEC;
        total_time += time_taken;
        printf("Time %i: %f\n", e, time_taken);
    }
    double avg_time = total_time/evals;
    printf("Avg. time: %f\n", avg_time);
    double tmp = (n/1000);
    double gf_ops = tmp*tmp*(2*n-1);
    printf("Mega-floating point operations: %f\n", gf_ops);
    double mflops = gf_ops/avg_time;
    printf("MFLOPS: %f\n", mflops);

    free(mat1);
    free(mat2);
    free(mat_out);

    return 0;
}
