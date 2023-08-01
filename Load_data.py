import os
import sys


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_genotype(file, path_plink = "/home/avillanu/plink/"):

    '''
    Function to load genotype matrix from ped or bed file (map and fam files needed).
    plink path should be specified
    '''
    

    os.system(f"{path_plink}plink --bfile {file} --recodeA --out {file} --noweb")
    os.system(f"cat {file}.raw | tail -n +2 | cut -d' ' -f7- > {file}.txt")


    X = np.loadtxt(f"{file}.txt", dtype="int")
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
    