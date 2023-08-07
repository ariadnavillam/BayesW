import numpy as np
from scipy import stats


def sigma_g_func(betas, pars):
    '''Defined posterior of sigma_g parameter'''
    betas_arr = np.array([beta.now for beta in betas])
    
    # betas_sqr = []
    # for k in range(1, pars["mixture_groups"]+1):
    #     b = betas_arr[pars["mixture_component"] == k]
    #     betas_sqr.append(b.T.dot(b))
    
    alpha = pars["alpha_sigma"] + 0.5 * (pars["n_markers"] - pars["v"][0] +1)
    beta = pars["beta_sigma"] + 0.5 * (pars["n_markers"] - pars["v"][0] +1)*np.linalg.norm(betas_arr, ord=2)**2
    
    return stats.invgamma(alpha, loc=0, scale=beta)

def log_beta(pars, epsilon_arr):
    '''computes the log-likelihood of the betas with dense data'''

    return lambda x: - pars["alpha"]*x*pars["sum_fail"]\
                    - np.sum(np.exp(pars["alpha"]*(epsilon_arr - pars["X_j"]*x) - np.euler_gamma)) \
                    - x**2/(2*pars["mixture_C"]*pars["sigma_g"])

def dev_log_beta(pars, epsilon_arr):
    '''computes the derivative of the log-likelihood of the betas with dense data
    '''
    return lambda x: - pars["alpha"]*pars["sum_fail"]\
                    + pars["alpha"]*np.sum(-pars["X_j"]*np.exp(pars["alpha"]*(epsilon_arr - pars["X_j"] * x) - np.euler_gamma)) \
                    - x**2/(2*pars["mixture_C"]*pars["sigma_g"])


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

    '''computes the log-likelihood of the mu parameter
    we added a term mu*alpha*d (mu is the current value) for being able to sample when alpha is high'''

    return lambda x : (- pars['alpha']*x*pars['d'] - \
                        np.sum(np.exp((epsilon_arr - x)*pars['alpha'] - np.euler_gamma)) \
                        - (x**2)/(2*pars["var_mu"]))

def dev_log_mu(pars, epsilon_arr):

    '''computes the derivative of the log-likelihood of the mu parameter'''

    return lambda x : (-pars['alpha']*pars['d'] + \
                       pars['alpha']*np.sum(np.exp((epsilon_arr - x)*pars['alpha'] -np.euler_gamma)) \
                        - x/pars["var_mu"])

