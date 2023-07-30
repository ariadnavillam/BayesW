#!/bin/bash


#SBATCH --ntasks 5

#SBATCH --cpus-per-task 4

#SBATCH --time 01:00:00

#SBATCH --mem-per-cpu=1gb

#SBATCH --output=/nfs/scistore18/bartogrp/avillanu/hydra/weibull_1000_200.log

module load gcc boost eigen openmpi

mkdir example/out_bayesW

srun src/hydra_G \
  --number-individuals 1000 \
  --number-markers 200 \
  --mpibayes bayesWMPI \
  --pheno example/weibull_1000_200.phen \
  --failure example/weibull_1000_200.fail \
  --quad_points 25 \
  --chain-length 500 \
  --thin 5 \
  --mcmc-out-dir example/out_bayesW \
  --mcmc-out-name weibull_1000_200 \
  --shuf-mark 1 \
  --sync-rate 1 \
  --bfile example/weibull_1000_200 \
  --S 0.001,0.01,0.1
  