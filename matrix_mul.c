#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>

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

int main() {
    int n = 256;
    printf("Time in [sec] for %ix%i:\n",n,n);
    int evals = 10;
    struct timeval start, end;
    long double avg_time = 0;

    float* mat1 = (float*) malloc(n*n*sizeof(float));
    float* mat2 = (float*) malloc(n*n*sizeof(float));
    float* mat_out = malloc(n*n*sizeof(float));

    
    for (int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            mat1[i*n+j] = (float)(i+j);
            mat2[i*n+j] = (float)(i+j);
        }
    }

    for (int e = 0; e < evals; e++) {
        // actual matrix multiplication
        gettimeofday(&start, NULL);
        normal_matrix_mul(mat1, mat2, mat_out, n);
        gettimeofday(&end, NULL);
        long long microseconds = ((end.tv_sec - start.tv_sec) * 1000*1000 + end.tv_usec - start.tv_usec);
        printf("Time %i in us: %lld\n", e, microseconds);
        avg_time += (long double)(microseconds)/evals;
    }
    printf("Avg. time: %Lf\n", avg_time);
    long long no_fops = n * n * (2 * n - 1);
    printf("floating point operations: %lld\n", no_fops);
    long double mega_flops = no_fops / avg_time;
    printf("MFLOPS: %Lf\n", mega_flops);

    free(mat1);
    free(mat2);
    free(mat_out);

    return 0;
}
