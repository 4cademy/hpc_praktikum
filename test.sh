#!/bin/bash

#use the according number to run the function in main
#define NORMAL 0
#define LOOP_UNROLLING 1
#define LOOP_SWAP 2
#define TILING 3



for i in 0 1 2 3
do
    echo "Running $i"
    ./matrix_mul $i >> matmul.txt 2>&1
done