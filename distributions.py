import numpy as np
from scipy import stats


def sigma_g_func(betas, pars):
    '''Defined posterior of sigma_g parameter'''
    betas_arr = np.array([beta.now for beta in betas])
    _, gamma_k = np.unique(pars["mixture_component"], return_counts=True)
    betas_sqr = []
    for k in range(1, pars["mixture_groups"]+1):
        b = betas_arr[pars["mixture_component"] == k]
        betas_sqr = b.T.dot(b)
    
    alpha = pars["alpha_sigma"] + 0.5 * np.sum(gamma_k)
    beta = pars["beta_sigma"] + 0.5 * np.sum(gamma_k*betas_sqr)
    
    return stats.invgamma(alpha, loc=0, scale=beta)

def log_beta(pars, partial_sums):
    '''computes the log-likelihood of the betas'''
    return lambda x: - pars["alpha"]*x*pars["sum_fail"]\
            + np.exp(pars["alpha"]*pars["mean_sd_ratio"]*x) \
            * (partial_sums[0] + partial_sums[1]*np.exp(-pars["alpha"]*x/pars["sd"]) + partial_sums[2]*np.exp(-2*pars["alpha"]*x/pars["sd"])) \
            - x**2/(2*pars["mixture_C"]*pars["sigma_g"])

def dev_log_beta(pars, partial_sums):
    '''computes the first derivative of the log-likelihood of the betas'''
    return lambda x: -pars["alpha"]*pars["sum_fail"] \
            + pars["alpha"]*pars["mean_sd_ratio"]*np.exp(pars["alpha"]*pars["mean_sd_ratio"]*x) \
            * (partial_sums[0] + partial_sums[1]*np.exp(-pars["alpha"]*x/pars["sd"]) + partial_sums[2]*np.exp(-2*pars["alpha"]*x/pars["sd"])) \
            +  np.exp(pars["alpha"]*pars["mean_sd_ratio"]*x) \
            * (partial_sums[1]*np.exp(-pars["alpha"]*x/pars["sd"])*(-pars["alpha"]/pars["sd"]) + partial_sums[2]*np.exp(-2*pars["alpha"]*x/pars["sd"])*(-2*pars["alpha"]/pars["sd"])) \
            - x/(pars["mixture_C"]*pars["sigma_g"])

def log_alpha(pars, epsilon_arr):
    '''computes the log-likelihood of the alpha parameter'''
    return lambda x: (pars["alpha_zero"] + pars["d"] -1)*np.log(x) \
            + x*(np.sum(epsilon_arr*pars["d_array"]) - pars["kappa_zero"]) \
            - np.sum(np.exp(epsilon_arr * x - np.euler_gamma))

def dev_log_alpha(pars, epsilon_arr):
    '''computes the first derivative of the log-likelihood of the alpha parameter'''
    return lambda x: (pars["alpha_zero"] + pars["d"] -1)/x \
            + np.sum(epsilon_arr*pars["d_array"]) - pars["kappa_zero"] \
            - np.sum(epsilon_arr * np.exp(epsilon_arr * x - np.euler_gamma))

def log_mu(pars, epsilon_arr):
    '''computes the log-likelihood of the mu parameter'''
    return lambda x : (-pars['alpha']*x*pars['d'] - \
                       np.sum(np.exp((epsilon_arr - x)*pars['alpha'] -np.euler_gamma)) \
                        - (x**2)/(2*pars["var_mu"]))

def dev_log_mu(pars, epsilon_arr):
    '''computes the derivative of the log-likelihood of the mu parameter'''
    return lambda x : (-pars['alpha']*pars['d'] + \
                       pars['alpha']*np.sum(np.exp((epsilon_arr - x)*pars['alpha'] -np.euler_gamma)) \
                        - x/pars["var_mu"])

