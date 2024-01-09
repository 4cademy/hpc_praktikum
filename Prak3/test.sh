#!/bin/bash

for i in 6000 7000 8000 9000 10000
do
    echo "running matrix Multiplication with size= $i"
    ./program $i >> results_prefetch.txt
done