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

void tiling_matrix_mul(const float* mat1, const float* mat2, float* mat_out, int n, int tiling_factor, int tile_size) {
    float* result_tile = malloc(tile_size*tile_size*sizeof(float));
    float* mat1_tile = malloc(tile_size*tile_size*sizeof(float));
    float* mat2_tile = malloc(tile_size*tile_size*sizeof(float));
    for (int tx_out = 0; tx_out < tiling_factor; tx_out++) {
        for (int ty_out = 0; ty_out < tiling_factor; ty_out++) {
            // copy tile from mat1
            for (int i = 0; i < tile_size; i++) {
                for (int j = 0; j < tile_size; j++) {
                    mat1_tile[i*tile_size+j] = mat1[(((ty_out*tile_size)+i)*n)+((0 * tile_size) + j)];
                }
            }

            // copy tile from mat2
            for (int i = 0; i < tile_size; i++) {
                for (int j = 0; j < tile_size; j++) {
                    mat2_tile[i*tile_size+j] = mat2[(((0 * tile_size) + i) * n) + ((tx_out * tile_size) + j)];
                }
            }
            // multiply tiles
            normal_matrix_mul(mat1_tile, mat2_tile, result_tile, tile_size);
            // copy result tile to mat_out
            for (int i = 0; i < tile_size; i++) {
                for (int j = 0; j < tile_size; j++) {
                    mat_out[(((ty_out*tile_size)+i)*n)+((tx_out * tile_size) + j)] += result_tile[i*tile_size+j];
                }
            }
            for (int tile_index = 1; tile_index < tiling_factor; tile_index++){
                // copy tile from mat1
                for (int i = 0; i < tile_size; i++) {
                    for (int j = 0; j < tile_size; j++) {
                        mat1_tile[i*tile_size+j] = mat1[(((ty_out*tile_size)+i)*n)+((tile_index * tile_size) + j)];
                    }
                }

                // copy tile from mat2
                for (int i = 0; i < tile_size; i++) {
                    for (int j = 0; j < tile_size; j++) {
                        mat2_tile[i*tile_size+j] = mat2[(((tile_index * tile_size) + i) * n) + ((tx_out * tile_size) + j)];
                    }
                }
                // multiply tiles
                normal_matrix_mul(mat1_tile, mat2_tile, result_tile, tile_size);
                // copy result tile to mat_out
                for (int i = 0; i < tile_size; i++) {
                    for (int j = 0; j < tile_size; j++) {
                        mat_out[(((ty_out*tile_size)+i)*n)+((tx_out * tile_size) + j)] += result_tile[i*tile_size+j];
                    }
                }
            }
        }
    }
    free(result_tile);
    free(mat1_tile);
    free(mat2_tile);
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
        // reset output matrix
        for (int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                mat_out[i*n+j] = 0;
            }
        }
        // actual matrix multiplication
        int tiling_factor = 16;
        int tile_size = n/tiling_factor;
        gettimeofday(&start, NULL);
        // tiling_matrix_mul(mat1, mat2, mat_out, n, tiling_factor, tile_size);
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
