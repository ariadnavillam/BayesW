#!/bin/bash

for ((n=0;n<10;n++))
do
    echo $n
    num=$(( $RANDOM % 50 + 1 ))
    echo $num

    python3 Simulate_data_files.py "$num"

    python3 BayesW_dense.py Weibull_dense_10000_10000_"$num"
done