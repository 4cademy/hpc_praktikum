#!/bin/bash
module purge
echo "module purge"
sleep 2
module switch modenv/ml
echo "module switch modenv/ml"
sleep 2
module load NVHPC
echo "module load NVHPC"
sleep 2
module load GCC/8.3.0
echo "module load GCC/8.3.0"
sleep 2
module load NVHPC
echo "module load NVHPC"
