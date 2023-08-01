import numpy as np
from scipy import stats
import helpers
import ars
import matplotlib.pyplot as plt
import Bayes_arms

class Parameter:
    def __init__(self, log_dens_f, dev_log_dens, bounds, xinit, init_value):
        self.current_value = init_value
        self.init_value = init_value
        self.previous_values = []
        self.f = log_dens_f
        self.df = dev_log_dens
        self.bounds = bounds
        self.now = init_value
        self.xinit = xinit

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
    
    def update(self, value):
        self.now  = value
        self.current_value
        self.previous_values.append(value)
        
    
    def sample_posterior(self, params, epsilon_or_sums, xinit = None, bounds = None, dometrop = 0):

        if bounds != None:
            self.bounds = bounds

        ninit = 4
        npoint = 100
        nsamp = 1
        ncent = 4
        convex = 1.0
        xprev = self.now
        xcent = qcent = [5.,30.,79.,95.]
        xl = self.bounds[0]
        xr = self.bounds[1]
        xsamp = []
        
        if xinit == None:
            xinit = [self.now * x for x in self.xinit]
        

        err = Bayes_arms.arms(xinit, ninit, xl,xr, self.f(params,epsilon_or_sums),
                              convex,npoint,dometrop,xprev,nsamp,qcent,xcent,ncent, xsamp)

        # bounds = self.get_bounds()
        # self.bounds_list.append(np.array(bounds))
        # self.previous_values.append(self.current_value)
         
        # err = ars.adaptive_rejection_sampling(x0=self.now*self.sampler_x0, 
        #                                               log_unnorm_prob=self.f(params, epsilon_or_sums), 
        #                                               derivative= self.df(params, epsilon_or_sums), 
        #                                               num_samples=n_samples, bounds=bounds, ddx=ddx)
        
        if err != 0:
            print("Error ", err, " in arms")
        else:      
            self.update(xsamp[0])

        return 
    
    def plot_sampled_values(self, truth = None):
        #bo = np.array(self.bounds_list)
        if truth != None:
            plt.axhline(y=truth, linestyle= "--", color="k")
        plt.plot(self.previous_values)
        #plt.fill_between(range(0,len(self.previous_values)), y1=bo[:,0], y2=bo[:,1], alpha=0.4)

class SimpleParameter():
    def __init__(self,  init_value ):
        self.current_value = init_value
        self.init_value = init_value
        self.previous_values = []
        self.now = init_value
        
    
    def sample_posterior(self, func):
        self.previous_values.append(self.now)
        x = func.rvs(size = 1)[0]
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
    mix = int(pars["mixture_component"][j])
    if mix != 0:
        pars['mixture_Ck'] = pars["Ck"][mix-1]

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

def init_parameters(n_markers, l_mix, data):
    '''
    Function to initilize the parameters dictionary
    
    Input: 
    - n_markers: number of markers
    - l_mix: number of mixture components
    - data: the simulated data

    Output:
    - pars: dictionary with all the paremeters
    - alpha_ini: the initial value of alpha
    - sigma_ini: the initial value of sigma
    
    '''
    markers, d_fail, y_data_log= data

    mu = np.mean(y_data_log)
    #alpha_ini = np.var(y_data_log)
    alpha_ini = np.pi/np.sqrt( 6 * np.sum( (y_data_log-mu) **2) / (len(y_data_log)-1))
    sigma_g_ini = np.pi**2/(6* n_markers * alpha_ini**2)
    norm_markers = helpers.normalize_markers(markers)
    pi_L = np.zeros(l_mix)
    pi_L[0] = 0.99
    pi_L[1:] = (1 - pi_L[0])/(l_mix - 1)
    
    pars = {"alpha": alpha_ini, 
            "sigma_g": sigma_g_ini,
            "d": np.sum(d_fail), 
            "mu": mu,
            "var_mu": 100, 
            "var_delta": 100,
            
            "alpha_zero": 0.01,
            "kappa_zero": 0.01,

            "d_array": d_fail,

            "alpha_sigma": 1,
            "beta_sigma": 0.0001,

            "n_markers": n_markers,
            

            # "mixture_Ck_all": np.ones(n_markers)*0.01/n_markers,
            # "mixture_groups": 1,
            "mixture_component": np.zeros(n_markers),
            
            "v": np.zeros(l_mix),

            "sum_fail_all": (d_fail * norm_markers.T).sum(axis=1),
            "mean_sd_ratio_all": np.mean(markers, axis=0)/np.std(markers, axis=0),
            "sd_all": np.std(markers, axis=0),


            "Ck": [0.1, 0.01, 0.001],
            "l_mix": l_mix,
            "pi_L": pi_L,
            "marginal_likelihoods": np.ones(l_mix),

            "hyperparameters": np.zeros(l_mix +1),
            "gammas": np.zeros(n_markers),

    }
    
    return pars, alpha_ini, sigma_g_ini


def simulate_data(mu_true, alpha_true, sigma_g_true, n_markers, n_samples, n_covs, prevalence):
    '''
    Function to simulate markers and phenotype, given the parameters.
    
    Input:
    - mu_true: intercept 
    - alpha_true: alpha 
    - sigma_g_true: true sigma_g value.
    - n_markers: number of markers
    - n_samples: number of samples
    - n_covs: number of covariates
    - prevalence: prevalence of the phenotype (how many people have had the event)

    Output:
    - markers: simulated marker data as a numpy array
    - betas: simulated beta coefficients as a numpy array
    - cov: simulated covariate data as a numpy array.
    - d_fail: simulated failure indicator (1: event ocurred, 0: it didnt) 
    - log_data: log phenotype data

    '''
    ## log Yi = mu + xB + wi/alpha + K/alpha

    ## Draw the errors from a standard Gumbel(wi)
    gumbel_dis = stats.gumbel_r(loc=0, scale=1)
    w = gumbel_dis.rvs(size=(n_samples,1))

    ## genetic effects
    betas = np.random.normal(0, np.sqrt(sigma_g_true/n_markers), size = (n_markers, 1))
    markers = np.random.binomial(2, 0.5, (n_samples, n_markers))

    g = helpers.normalize_markers(markers).dot(betas)

    cov = np.random.binomial(1, 0.5, size = (n_samples, n_covs)) #z matrix of 

    ## failure indicator vector
    d_fail = np.random.choice([0,1], p=[1-prevalence, prevalence],size = n_samples)

    log_data = mu_true + g + w/alpha_true + np.euler_gamma/alpha_true
    log_data = log_data.reshape(log_data.shape[0])

    return (markers, betas, cov, d_fail, log_data)


def gh_integrand_adaptive(s, pars, exp_epsilon, marker):
    """
    Computes the value of an integrand function.

    Input:
    - pars: parameters dictionary
    - exp_epsilon: exponentiated adjusted residuals
    - s: quadrature point, roots of the Hermite polynomial
    - marker: marker j (column of the markers matrix)

    Output:
    - np.exp(temp): computed value of the integrand function
    """
    
    # vi is a vector of exp(vi) # exp epsilon function
    
    # Calculate the exponent argument (sparse)
    # temp = -pars["alpha"] * pars["s"] * pars["dj"] * pars["sqrt_2Ck_sigmaG"] + partial_sums.sum() \
    #     - np.exp(pars["alpha"] * pars["mean_sd_ratio"] * pars["s"] * pars["sqrt_2Ck_sigmaG"]) \
    #     * ( partial_sums[0] + partial_sums[1] * np.exp(-pars["alpha"] * pars["s"] * pars["sqrt_2Ck_sigmaG"] / pars["sd"])\
    #      + partial * np.exp(-2 * alpha * s * sqrt_2Ck_sigmaG / sd)
    # ) - s ** 2

    temp = -pars["alpha"] * s * pars["sum_fail"] * pars["sqrt_2Ck_sigmaG"] \
            + np.sum((exp_epsilon * (1 -  np.exp(-marker*s*pars["sqrt_2Ck_sigmaG"]*pars["alpha"]) ) )) \
            - s ** 2
    
    
    return np.exp(temp)

def gauss_hermite_adaptive_integral(k, exp_epsilon, pars, m_points, marker):
    '''
    Compute the value of the integral using Adaptive Gauss-hermite quadrture

    Input:
    - k: mixture component
    - exp_epsilon: exponentiated epsilon
    - pars: parameter dictionary
    - m_points: number of quadrature points
    - marker: marker j (column of the markers matrix)

    Output:
    - pars["sigma"]*temp: value of the integral
    '''

    pars["sqrt_2Ck_sigmaG"] = np.sqrt(2 * pars["Ck"][k] * pars["sigma_g"])
    x_points = np.zeros(m_points - 1)
    w_weights = np.zeros(m_points)

    if m_points == 3:
        

        x_points[0] = 1.2247448713916
        x_points[1] = - x_points[0]

        w_weights[0] = 1.3239311752136
        w_weights[1] = w_weights[0]

        

        w_weights[2] = 1.1816359006037

        x_points = pars["sigma"] * x_points
        temp = np.sum([w_weights[i] * gh_integrand_adaptive(x_points[i], pars, exp_epsilon, marker) for i in range(0,m_points - 1)])
        temp += w_weights[-1]
    
    elif m_points == 5:

        x_points[np.arange(0,m_points-1,2)] = [2.0201828704561, 0.95857246461382]
        x_points[np.arange(1,m_points-1,2)] =  - x_points[np.arange(0,m_points-1,2)]

        w_weights[np.arange(0,m_points,2)] = [1.181488625536, 0.98658099675143, 0.94530872048294]
        w_weights[np.arange(1,m_points, 2)] = w_weights[np.arange(0, m_points - 1, 2)]

        x_points = pars["sigma"] * x_points
        temp = np.sum([w_weights[i] * gh_integrand_adaptive(x_points[i], pars, exp_epsilon, marker) for i in range(0,m_points - 1)])
        temp += w_weights[-1]

    elif m_points == 7:

        x_points[np.arange(0,m_points-1,2)] = [2.6519613568352, 1.6735516287675, 0.81628788285897]
        x_points[np.arange(1,m_points-1,2)] =  - x_points[np.arange(0,m_points-1,2)]

        w_weights[np.arange(0,m_points,2)] = [1.1013307296103, 0.8971846002252, 0.8286873032836, 0.81026461755681]
        w_weights[np.arange(1,m_points, 2)] = w_weights[np.arange(0, m_points - 1, 2)]

        x_points = pars["sigma"] * x_points
        temp = np.sum([w_weights[i] * gh_integrand_adaptive(x_points[i], pars, exp_epsilon, marker) for i in range(0,m_points - 1)])
        temp += w_weights[-1]
    

    else:
        raise ValueError('Possible number of quad points: 3, 5, 7 ')



    return pars["sigma"]*temp

def marginal_likelihood_vec_calc(pars, exp_epsilon, n, marker):
    '''Calculate the marginal likelihood vectors for components not equal to zero
    Input:
    - pars: parameters dictionary
    - exp_epsilon: exponentiated residuals
    - n: number of quadrature points
    - marker: column of the markers matrix
    
    Ouput:
    - pars["marginal_likelihoods"]: vector with probabilities for each mixture'''
    
    exp_sum = np.sum(exp_epsilon * marker * marker)

    for k in range(len(pars["Ck"])):
        pars["sigma"] = 1/np.sqrt(1 + pars["alpha"]**2 * pars["sigma_g"] * pars["Ck"][k] * exp_sum)
        temp = pars["pi_L"][k+1] \
            * gauss_hermite_adaptive_integral(k=k, exp_epsilon=exp_epsilon, pars=pars, m_points = n, marker=marker )

        if np.isinf(temp):
            print(k, exp_sum, pars["pi_L"][k+1], gauss_hermite_adaptive_integral(k=k, exp_epsilon=exp_epsilon, pars=pars, m_points = n, marker=marker ))
       
        pars["marginal_likelihoods"][k+1] = temp

    return


    
if __name__ == "__main__":
    pass
    