import os
import sys

import numpy as np
from scipy import stats
import pandas as pd
from Load_data import *

#### ARGUMENTS
dataset = "Weibull"
type_marker = "sparse"
N = 5000
M = 10000
causal = 200

path = "files_sim"

mu = 4.1

alpha = 10
sigma_g = np.pi**2/(6*alpha**2)

####

if path.endswith("/") == False:
    path = path + "/"

dataset = f"{dataset}_{type_marker}_{N}_{M}"

np.random.seed(1)

b = np.random.normal(0, np.sqrt(sigma_g/causal), size = causal)

gen_file = "/home/avillanu/BayesW_data_sim/t_M10K_N_5K"
markers = load_genotype(gen_file)

# if type_marker == "dense":
#     markers = np.random.normal(0, 1, (N, M))
# elif type_marker =="sparse":
#     markers = np.random.binomial(2, 0.5, (N, M))
# else:
#     print("Wrong marker type. Options: dense, sparse.")
#     exit()

beta = np.zeros(M)
index = np.random.choice(np.arange(0,M), causal)

h2 = 0.5

beta[index] = b

g = markers.dot(beta)

# gumbel_dis = stats.gumbel_r(loc=0, scale=1)
# w = gumbel_dis.rvs(size=(N,1))

w = -np.log(-np.log(1-np.random.uniform(size=N)))
log_data = mu + g + w/alpha + np.euler_gamma/alpha
## write hyperparameters file

censoring_time = np.random.uniform(600,1000, size=N)

cens = np.log(censoring_time)


isFailure = log_data <= cens
y_log = np.where(isFailure , log_data, cens)

var_g = np.sum(beta**2)

print(var_g)

print(f'''h2\t{var_g/(var_g + sigma_g)}
var_g\t{var_g}
mu\t{mu}
alpha\t{alpha} ''',
file=open(path + dataset +'.h2', "w"))

## write beta file
pd.DataFrame({"index":index, "effect": b}).to_csv(path + dataset + ".beta", index=False, sep='\t', header=None)

## failure indicator vector
d_fail = isFailure * 1


plink_q = input("Create plink files? (Y/n)")

if plink_q != "n":

    ped = pd.DataFrame({"FID" : np.arange(1, N+1, dtype="int"),
                    "IID" : np.arange(1, N+1, dtype="int"),
                    "PID" : np.zeros(N,dtype='int'),
                    "MID" : np.zeros(N,dtype='int'),
                    "Sex" : np.zeros(N,dtype='int'),
                    "phen" : np.zeros(N,dtype='int') })

    markersdf = pd.DataFrame(markers)
    di = {0: "AA", 1:"CA", 2: "CC"}
    df = markersdf.replace(di)
    df['concat'] = pd.Series(df.fillna('').values.tolist()).str.join('')
    seq = df['concat'].to_numpy()
    genotype = pd.DataFrame(np.array([np.array(list(s)) for s in seq]))


    # for col in df.columns:
    #     # Split letters into two columns
    #     df[[str(col) + '_Letter1', str(col) + '_Letter2']] = df[col].str.split('', expand=True).iloc[:, 1:3]

    #     # Drop the original column
    #     df = df.drop(col, axis=1)

    pd.concat([ped, genotype], axis=1).to_csv(path + dataset + ".ped", index=False, sep='\t', header=None)

    map = pd.DataFrame({"chr" : np.ones(M, dtype="int"),
                    "rs" : ["rs" + str(i) for i in range(1, M+1)],
                    "dist" : np.zeros(M, dtype="int"),
                    "bp" : np.arange(1,M+1, dtype="int")})

    map.to_csv(path + dataset + ".map", index=False, sep='\t', header=None)

    fam = pd.DataFrame({"FID" : np.arange(1, N+1, dtype="int"),
                  "IID" : np.arange(1, N+1, dtype="int"),
                  "IDfather" : np.zeros(N,dtype='int'),
                  "IDmother" : np.zeros(N,dtype='int'),
                  "Sex" : np.zeros(N,dtype='int'),
                  "phen" : np.repeat(-9,N) })

    fam.to_csv(path + dataset + ".fam", index=False, sep='\t', header=None)

    os.system(f"/home/avillanu/plink/plink --file {path + dataset} --make-bed --out {path+dataset} --noweb")

else:
    ## create raw matrix and binary file
    np.savetxt(path + dataset + ".txt", markers)
    np.save(path + dataset + ".npy", markers)


phen = pd.DataFrame({"FID" : np.arange(1, N+1, dtype="int"),
                "IID" : np.arange(1, N+1, dtype="int"),
                "phen" : y_log })

phen.to_csv(path + dataset + ".phen", index=False, sep=' ', header=None)

np.savetxt(path + dataset + ".fail", d_fail, '%1i')


# bash_script = f"""#!/bin/bash


# #SBATCH --ntasks 5

# #SBATCH --cpus-per-task 4

# #SBATCH --time 01:00:00

# #SBATCH --mem-per-cpu=1gb

# #SBATCH --output=/nfs/scistore13/rbngrp/avillanu/hydra/{dataset}.log

# module load gcc boost eigen openmpi

# mkdir example/out_bayesW

# srun src/hydra_G \\
#   --number-individuals {N} \\
#   --number-markers {M} \\
#   --mpibayes bayesWMPI \\
#   --pheno example/{dataset}.phen \\
#   --failure example/{dataset}.fail \\
#   --quad_points 25 \\
#   --chain-length 500 \\
#   --thin 5 \\
#   --mcmc-out-dir example/out_bayesW \\
#   --mcmc-out-name {dataset} \\
#   --shuf-mark 1 \\
#   --sync-rate 1 \\
#   --bfile example/{dataset} \\
#   --S 0.001,0.01,0.1
#   """

# with open("run_bayes.sh","w+") as f:
#     f.writelines(bash_script)



# move = input("Move new files to the cluster? (Y/n)")

# if move != "n":

#     os.system(f"scp run_bayes.sh avillanu@bea81.hpc.ista.ac.at:/nfs/scistore18/bartogrp/avillanu/hydra/.")

#     os.system(f"scp {path + dataset}.* avillanu@bea81.hpc.ista.ac.at:/nfs/scistore18/bartogrp/avillanu/hydra/example/.")