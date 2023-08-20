import os
import sys


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_betas(file, M):
    '''
    Function to read the true betas file

    Input:
    - file: file where the betas were saved (.beta)
    - M: total number of markers
    '''

    betas = pd.read_table(file, header=None)
    index = betas[0].to_numpy()
    b = betas[1].to_numpy()
    betas = np.zeros(M)
    betas[index] = b
    return betas

def load_genotype(file, geno_type = "dense", path_plink = "/home/avillanu/plink/"):

    '''
    Function to load genotype matrix from ped or bed file (map and fam files needed).
    plink path should be specified
    '''
    
    if geno_type == "sparse":
        if os.path.isfile(file+".raw") == False:
            os.system(f"{path_plink}plink --bfile {file} --recodeA --out {file} --noweb")
            
        os.system(f"cat {file}.raw | tail -n +2 | cut -d' ' -f7- > {file}.txt")

        X = np.loadtxt(f"{file}.txt", dtype="int")
    
    elif geno_type == "binary":
        X = np.load(f"{file}.npy")
    
    elif geno_type =="dense":
        X = np.loadtxt(f"{file}.txt")
    
    else:
        print("Enter correct type of markers. Options: dense | binary | sparse.")
        exit()

    return X

def load_fail(file):

    '''
    Read failure file (one column file)
    '''
    fail = np.loadtxt(file, dtype="int")
    return fail

def load_phen(file):
    '''
    Read phenotype file. (IID, FID, phenotype)
    '''
    df = pd.read_table(file, header=None, sep=" ")
    phen = df[2].to_numpy()
    return phen



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("File name missing.")
        exit(1)

    gen_file = sys.argv[1]
    fail_file = sys.argv[2]
    phen_file = sys.argv[3]
    X = load_genotype(gen_file)
    d_array = load_fail(fail_file)
    data = load_phen(phen_file)
    