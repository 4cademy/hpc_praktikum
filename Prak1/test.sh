#!/bin/bash

# use the according number to run the function in main
# define NORMAL 0
# define LOOP_UNROLLING 1
# define LOOP_SWAP 2
# define TILING 3

for j in 128 256 512 1024 2048
do
  for i in 0 1 2 3
  do
      echo "Running $i $j "
      ./cmake-build-debug/hpc_praktikum $i $j >> matmul.txt 2>&1
  done
done