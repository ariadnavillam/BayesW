import numpy as np
from scipy import stats
import helpers
import ars
import matplotlib.pyplot as plt

class Parameter:
    def __init__(self, log_dens_f, dev_log_dens, bounds, init_value, sampler_x0 = 1):
        self.current_value = init_value
        self.init_value = init_value
        self.previous_values = []
        self.f = log_dens_f
        self.df = dev_log_dens
        self.bounds = bounds
        self.now = init_value
        self.sampler_x0 = sampler_x0
        self.bounds_list = []

    def get_bounds(self):
        
        val_bounds = self.bounds
        bounds = (val_bounds[0]*self.current_value, val_bounds[1]*self.current_value)

        if helpers.bounds_error(bounds) or self.now == 0:
            return (self.bounds[0] - 1, self.bounds[1] - 1)
        elif bounds[1] < bounds [0]:
            return helpers.change_order(bounds)
        else:
            return bounds
    
    def get_value(self):
        return self.current_value
    
    def sample_posterior(self, params, epsilon_or_sums, n_samples=1):

        bounds = self.get_bounds()
        self.bounds_list.append(np.array(bounds))
        self.previous_values.append(self.current_value)
        try: 
            samples, xs = ars.adaptive_rejection_sampling(x0=self.now*self.sampler_x0, 
                                                      log_unnorm_prob=self.f(params, epsilon_or_sums), 
                                                      derivative= self.df(params, epsilon_or_sums), 
                                                      num_samples=n_samples, bounds=bounds)
        except:
            print(self.now, bounds)
            raise ValueError("Something in the sampler went wrong")
        
        if n_samples == 1:
            self.current_value = samples[0]
        else:
            self.current_value = samples
        self.now = self.current_value
        return self.current_value
    
    def plot_sampled_values(self, truth = None):
        bo = np.array(self.bounds_list)
        if truth != None:
            plt.axhline(y=truth, linestyle= "--", color="k")
        plt.plot(self.previous_values)
        plt.fill_between(range(0,len(self.previous_values)), y1=bo[:,0], y2=bo[:,1], alpha=0.4)

class SimpleParameter():
    def __init__(self,  init_value ):
        self.current_value = init_value
        self.init_value = init_value
        self.previous_values = []
        self.now = init_value
        
    
    def sample_posterior(self, func):
        x = func.rvs(size = 1)[0]
        self.previous_values.append(x)
        self.now = x
        return x
    
    def plot_sampled_values(self, truth = None):
        if truth != None:
            plt.axhline(y=truth, linestyle= "--", color="k")
        plt.plot(self.previous_values)




def prepare_pars_for_beta(pars,j):
    pars['mean_sd_ratio'] = pars['mean_sd_ratio_all'][j]
    pars['sd'] = pars['sd_all'][j]
    pars['sum_fail'] = pars['sum_fail_all'][j]
    pars['mixture_C'] = pars['mixture_C_all'][j]
    return pars

def calculate_exp_epsilon(pars, epsilon):
    return np.exp(pars["alpha"]*epsilon - np.euler_gamma)

def compute_partial_sums(exp_epsilon, marker):
    total_sum = exp_epsilon.sum()
    partial_sums = np.zeros(3)
    for i in [2,1]:
        idx = np.argwhere(marker == i)
        partial_sums[i] = exp_epsilon[idx].sum()
    
    partial_sums[0] = total_sum - partial_sums.sum()

    return partial_sums

def init_parameters(n_markers, n_samples, n_covs, l_mix, data):

    markers, _, _, d_fail, y_data_log= data

    mu = np.mean(y_data_log)
    alpha_ini = np.var(y_data_log)
    #alpha_ini = np.pi/np.sqrt( 6*np.sum( (y_data_log-mu) **2) / (len(y_data_log)-1))
    sigma_g_ini = np.pi**2/(6*alpha_ini**2)/l_mix

    
    
    pars = {"alpha": alpha_ini, 
            "sigma_g": sigma_g_ini,
            "d": np.sum(d_fail), 

            "var_mu": 100, 
            "var_delta": 100,
            
            "alpha_zero": 0.01,
            "kappa_zero": 0.01,

            "d_array": d_fail,

            "alpha_sigma": 1,
            "beta_sigma": 0.0001,

            "mixture_C_all": np.ones(n_markers)*1/n_markers,
            "mixture_groups": 1,
            "mixture_component": np.ones(n_markers),

            "sum_fail_all": (d_fail * markers.T).sum(axis=1),
            "mean_sd_ratio_all": np.mean(markers, axis=0)/np.std(markers, axis=0),
            "sd_all": np.std(markers, axis=0),
            "pi_vector": np.zeros(l_mix) + 1,
            "hyperparameters": np.zeros(l_mix +1),
            "gammas": np.zeros(n_markers),

    }
    
    return pars, alpha_ini, sigma_g_ini


def simulate_data(mu_true, alpha_true, sigma_g_true, n_markers, n_samples, n_covs):
    gumbel_dis = stats.gumbel_r(loc=0, scale=1)
    w = gumbel_dis.rvs(size=(n_samples,1))
    betas = np.random.normal(0, np.sqrt(1/n_markers), size = (n_markers, 1))
    markers = np.random.binomial(2, sigma_g_true, (n_samples, n_markers))
    g = markers.dot(betas)

    cov = np.random.binomial(1, 0.5, size = (n_samples, n_covs)) #z matrix of 
    d_fail = np.ones(n_samples)

    log_data = mu_true + g + w/alpha_true + np.euler_gamma/alpha_true
    log_data = log_data.reshape(log_data.shape[0])

    return (markers, betas, cov, d_fail, log_data)
    
if __name__ == "__main__":
    pass
    