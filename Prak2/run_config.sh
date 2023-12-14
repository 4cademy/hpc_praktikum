#!/bin/bash



read -p "Matrix Größe: " size
read -p "Anzahl Prozesse: " processes
read -p "Anzahl Knoten: " cores
read -p "Zeit angeben im Format: 00:00:00 " time

echo "Running matrix multiplication size= $size with $processes processes"
echo "Knoten $cores"
echo "Zeit beantragt: $time"

echo "Prozesse: $processes" >> matmul_$size.txt
echo "Knoten: $cores" >> matmul_$size.txt
srun -n $processes -N $cores -t time ./matrix_mul $size >> matmul_$size.txt 2>&1


