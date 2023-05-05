import numpy as np

def log_mu(pars, epsilon_arr):
    '''computes the log-likelihood of the mu parameters'''
    return lambda x : (-pars['alpha']*x*pars['d'] - np.sum(np.exp((epsilon_arr - x)*pars['alpha'] -np.euler_gamma)) - (x**2)/(2*pars["var_mu"]))

def dev_log_mu(pars, epsilon_arr):
    '''computes the derivative of the log-likelihood of the mu parameters'''
    return lambda x : (-pars['alpha']*pars['d'] + pars['alpha']*np.sum(np.exp((epsilon_arr - x)*pars['alpha'] -np.euler_gamma)) - x/pars["var_mu"])

