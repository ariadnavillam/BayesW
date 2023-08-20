import os
import time
import sys

import matplotlib.pyplot as plt

import numpy as np
from scipy import stats
from numpy.linalg import norm

### plots formatting
from matplotlib import rc
from matplotlib.colors import LinearSegmentedColormap
from cycler import cycler

from BayesW_arms_dev import *


rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Palatino'], 'size':14})
rc('text', usetex=True)
plt.rc('axes', prop_cycle=(cycler('color', ['red', 'gray', 'black', 'blue', 'green'])))#, )))

#import ars
import Bayes_arms

from BayesW_utils_dense import *
from Distributions_dense import *
from Load_data import *

## parameters

maxit = 100
quad_points = 7
l_mix = 4

name = sys.argv[1]
file_dir = "files_sim"

check_file = "checking.txt"
type_marker = "binary"
gen_file = f"{file_dir}/{name}"
fail_file = f"{file_dir}/{name}.fail"
phen_file = f"{file_dir}/{name}.phen"
betas_file = f"{file_dir}/{name}.beta"

hpars_file = f"{file_dir}/{name}.h2"

if os.path.isfile(gen_file+".h2"):
    hp = pd.read_table(gen_file + ".h2", header=None)
    h2 = hp[hp[0] =="h2"][1].to_numpy()[0]
    alpha_true = hp[hp[0] =="alpha"][1].to_numpy()[0]
    mu_true = hp[hp[0] =="mu"][1].to_numpy()[0]
    sigma_g_true = hp[hp[0] =="var_g"][1].to_numpy()[0]

## load data

start = time.time()
markers = load_genotype(gen_file, geno_type=type_marker)
d_fail = load_fail(fail_file)
y_data_log = load_phen(phen_file)

n_markers = markers.shape[1]
true_betas = get_betas(betas_file, n_markers)

stop1 = time.time()

print(f"Data loaded. {stop1-start} seconds.")
print(markers.shape)
## simulate data

# n_markers, n_samples,n_covs = 150,100,10

# mu_true = 3.9
# alpha_true = 10
# sigma_g_true = np.pi**2/(6*alpha_true**2)

# data = simulate_data(mu_true = mu_true, 
#                      alpha_true = alpha_true, 
#                      sigma_g_true = sigma_g_true,
#                      n_markers = n_markers,
#                      n_samples = n_samples,
#                      n_covs = n_covs,
#                      prevalence = 1
#                      )


# markers, betas_true, cov, d_fail, y_data_log = data

out_file = f"out_files/BayesW_out_{name}.tsv"

header = ["iter", "mu", "sigmaG", "alpha", "h2", "num_markers", "num_groups", "num_mixtures"]
print("\t".join(header), file=open(out_file, 'w'))

norm_markers = helpers.normalize_markers(markers)

pars, alpha_ini, sigma_g_ini = init_parameters(n_markers = n_markers,
                                                l_mix = l_mix, 
                                                data = (markers, d_fail, y_data_log) )

stop2 = time.time()
epsilon = y_data_log
mu = 4.1
xinit = np.array([0.995, 1, 1.005, 1.01]) * 4.1
log_unnorm_prob = log_mu(pars, epsilon)
derivative = dev_log_mu(pars, epsilon)

npoints = 10
xl = 2
xr = 10
ninit = 4
npoint = 100
nsamp = 1

times = []
samples = []
plt.figure(figsize=(10,6))
for p in [10,20,30,50, 60, 70, 80,90,100]:
    start = time.time()
    nsamp = 10
    xsamp = adaptive_rejection_sampling(xinit, ninit, xl, xr, log_unnorm_prob, derivative, p, nsamp)
    
    stop = time.time()
    xsamp = np.array(xsamp)
    plt.plot(p, xsamp.mean(), "ro")
    plt.errorbar(p, xsamp.mean(), yerr = xsamp.std(), capsize=10)
    times.append(stop-start)
    


plt.show()
print(times)



