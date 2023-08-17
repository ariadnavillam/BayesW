#!/usr/bin/env python3

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
n_markers = 10000

name = "Weibull_dense_10000_10000_99"#sys.argv[1]
file_dir = "files_sim"

check_file = "checking.txt"
type_marker = "dense"
gen_file = f"{file_dir}/{name}"
fail_file = f"{file_dir}/{name}.fail"
phen_file = f"{file_dir}/{name}.phen"

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


mix_comp = []
mu = Parameter(log_mu, dev_log_mu, 
               bounds = (2,10), 
               init_value = np.mean(y_data_log),
               xinit = [0.995, 1, 1.005, 1.01])

alpha = Parameter(log_alpha, dev_log_alpha,
                  bounds = (0, 400),
                  init_value= pars["alpha"],
                  xinit = [0.5, 1, 1.5, 3])

betas = list(np.zeros(n_markers))
for j in range(n_markers):
    betas[j] = Parameter(log_beta, dev_log_beta,
                bounds = (-1,1),
                init_value = 0, xinit= [1,1,1,1])

sigma_g = SimpleParameter(pars["sigma_g"])
    
#mu.now = mu_true
#alpha.now = alpha_true

epsilon = y_data_log - mu.now
np.set_printoptions(precision=3)



print("\t".join(["0", str(mu.now), str(sigma_g.now), str(alpha.now), "h2", "0", "0", "0"]), file=open(out_file, 'a'))

stop2 = time.time()

print(f"Initialization completed. {stop2 - stop1} seconds.")

markerI = np.arange(0,n_markers)

for it in range(maxit):
    
    #clear_output(wait=False)
    print('it: {}/{}'.format(it+1,maxit))

    ## SAMPLE MU
    pars["mu"] = mu.now
    epsilon = epsilon + mu.now
    x = mu.sample_posterior(pars, epsilon, bounds = (0.8*mu.now, 1.2*mu.now))
    
    epsilon = epsilon - mu.now

    ## SAMPLE ALPHA
    x = alpha.sample_posterior(pars, epsilon)
    pars["alpha"] = alpha.now
    
    
    vi = np.exp(pars["alpha"] * epsilon - np.euler_gamma)

    pars["marginal_likelihoods"][0] = pars["pi_L"][0] * np.sqrt(np.pi)
    pars["v"] = np.ones(pars["l_mix"])
    
    np.random.shuffle(markerI)
    for j in markerI:
        
        pars = prepare_pars_for_beta(pars, j)
        pars["X_j"] = norm_markers[:,j].flatten()
        
        beta = betas[j]

        if beta.now != 0:
            epsilon = epsilon + pars["X_j"]*beta.now
            vi = np.exp(pars["alpha"] * epsilon - np.euler_gamma)
        
        p_uni = np.random.uniform()

        pars["marginal_likelihoods"] = marginal_likelihood_vec_calc(pars, vi, quad_points)
        
        prob_acum = pars["marginal_likelihoods"][0]/pars["marginal_likelihoods"].sum()
        
        for k in range(pars["l_mix"]):
            if p_uni <= prob_acum:

                if k == 0:
                    beta.update(0)
                    pars["v"][k] += 1
                    pars["mixture_component"][j] = k
                
                else:
                    pars["mixture_C"] = pars["Ck"][k-1]
                    safe_limit = helpers.calculate_safe_limit(pars)

                    beta.sample_posterior(pars, epsilon, 
                                          bounds = (beta.now - safe_limit, beta.now + safe_limit),
                                          xinit = [beta.now - safe_limit/10 , beta.now,  beta.now + safe_limit/20, beta.now + safe_limit/10],
                                          dometrop = 0)

                    pars["v"][k] +=1
                    pars["mixture_component"][j] = k

                    epsilon = epsilon - pars["X_j"]*beta.now
                    vi = np.exp(pars["alpha"] * epsilon - np.euler_gamma)
                
                break

            else:
                if k + 1 == pars["l_mix"] - 1:
                    prob_acum = 1
                else:
                    prob_acum += pars["marginal_likelihoods"][k+1]/pars["marginal_likelihoods"].sum()
                    
    
    sigma_g.sample_posterior(sigma_g_func(betas, pars))
    pars["sigma_g"] = sigma_g.now

    print(sigma_g.now)

    pars["pi_L"] = np.random.dirichlet(pars["v"])
    pars["sqrt_2Ck_sigmaG"] = np.sqrt(2*pars["sigma_g"])

    mix_comp.append(pars["v"])

    
    print("\t".join([str(it+1), str(mu.now), str(sigma_g.now), str(alpha.now), "h2", str(np.sum(pars["v"][1:])), "1", str(len(pars["v"]))]), file=open(out_file, 'a'))

stop = time.time()
print(f"Finished! Total time: {stop - start} seconds.")
np.savetxt(f"out_files/betas_out_{name}.txt", np.array([beta.now for beta in betas]), fmt='%1.4f')

f = plt.figure(1, figsize=(10,8))
plt.subplot(2,2,2)
alpha.plot_sampled_values(truth = alpha_true)
plt.title(r"Weibull shape parameter: $\alpha$")

plt.subplot(2,2,1)
mu.plot_sampled_values(truth = mu_true)
plt.title(r"$\mu$")

plt.subplot(2,2,3)
for beta in betas:
    beta.plot_sampled_values()
plt.title(r"$\beta$")

plt.subplot(2,2,4)
sigma_g.plot_sampled_values(truth = sigma_g_true)
plt.title(r"$\sigma_G^2$")
plt.savefig(f"out_plots/Estimates_{name}.png", dpi=300)

mix_comp = np.array(mix_comp)
mix_comp = mix_comp/mix_comp.sum(axis=1).reshape(maxit,1)
mix_prop = np.cumsum(mix_comp, axis=1)
x = np.arange(maxit)
g = plt.figure(2)
plt.fill_between(x, np.repeat(0,maxit), mix_prop[:, 0], color='blue', alpha=0.3, label='k = 0')
plt.fill_between(x, mix_prop[:, 0], mix_prop[:, 1], color='green', alpha=0.3, label='k=1')
plt.fill_between(x, mix_prop[:, 1], mix_prop[:, 2], color='red', alpha=0.3, label='k=2')
plt.fill_between(x, mix_prop[:, 2], mix_prop[:, 3], color='orange', alpha=0.3, label='k=3')
plt.legend()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel("iteration")
plt.ylabel("Proportion of markers in component")
plt.tight_layout()
plt.savefig(f"out_plots/Mixtures_{name}.png", dpi=300)
