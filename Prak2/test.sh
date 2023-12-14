#!/bin/bash

# use the according number to run the function in main
# first argument of matrix

# k = Anzahl der Kerne
for k in 1 2
do
  # j = Matrixgröße
  for j in 2048
  do
    # i = Anzahl der parallelen Prozesse
    for i in 1 2 4 8 16
    do
      echo "Running matrix multiplication size= $j with $i processes"
      echo "Prozesse: $i" >> matmul_$j.txt
      echo "Kerne: $k" >> matmul_$j.txt
      srun -n $i -N $k -t 00:10:00 ./matrix_mul $j >> matmul_$j.txt 2>&1
    done
  done

  # j = Matrixgröße
  for j in 4096 8192 16384
  do
    # i = Anzahl der parallelen Prozesse
    for i in 32 48 64 80 96 104
    do
      echo "Running matrix multiplication size= $j with $i processes"
      echo "Prozesse: $i" >> matmul_$j.txt
      echo "Kerne: $k" >> matmul_$j.txt
      srun -n $i -N $k -t 00:10:00 ./matrix_mul $j >> matmul_$j.txt 2>&1
    done
  done
done