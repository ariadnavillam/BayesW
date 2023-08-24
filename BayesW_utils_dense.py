import numpy as np

import matplotlib.pyplot as plt
import BayesW_arms
import helpers

class Parameter:
    '''
    Class to store the parameters with unknown posterior distributions 
    '''

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
        
    
    def sample_posterior(self, params, epsilon, xinit = None, bounds = None, dometrop = 0):

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
        

        err = BayesW_arms.arms(xinit, ninit, xl, xr, self.f(params,epsilon),
                              convex,npoint,dometrop,xprev,nsamp,qcent,xcent,ncent, xsamp)
        
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
    '''
    Class to store parameters that have known posteriors
    '''
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
    '''
    Function to change the parameter values for each marker
    '''

    pars['sum_fail'] = pars['sum_fail_all'][j]
    pars['Xj_sqrt'] = pars['Xj_sqrt_all'][j]
    mix = int(pars["mixture_component"][j])
    if mix != 0:
        pars['mixture_Ck'] = pars["Ck"][mix-1]

    return pars

def calculate_exp_epsilon(pars, epsilon):
    return np.exp(pars["alpha"]*epsilon - np.euler_gamma)



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
    alpha_ini = np.pi/np.sqrt( 6 * np.sum( (y_data_log-mu) **2) / (len(y_data_log)-1))
    sigma_g_ini = np.pi**2/(6* n_markers * alpha_ini**2)

    norm_markers = helpers.normalize_markers(markers)

    pi_L = np.zeros(l_mix)
    pi_L[0] = 0.99
    pi_L[1:] = (1 - pi_L[0])/(l_mix - 1)
    
    markers_sqr = [] #np.diag(np.dot(norm_markers.T, norm_markers))
    sum_fail = [] #np.dot(norm_markers.T, d_fail).flatten()
    
    for j in range(n_markers):
        sum_fail.append(np.sum(norm_markers[:,j] * d_fail))
        markers_sqr.append(norm_markers[:,j] * norm_markers[:,j])

    
    
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
            

            "mixture_component": np.zeros(n_markers),
            
            "v": np.zeros(l_mix),



            "sum_fail_all": np.array(sum_fail), #(d_fail * norm_markers.T).sum(axis=1),
    


            "Ck": [0.001, 0.01, 0.1],
            "l_mix": l_mix,
            "pi_L": pi_L,
            "marginal_likelihoods": np.ones(l_mix),

            "Xj_sqrt_all": markers_sqr
    }
    
    return pars, alpha_ini, sigma_g_ini, 


def gh_integrand_adaptive(s, pars, exp_epsilon):
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
    alpha = pars["alpha"]
    sum_fail = pars["sum_fail"]

    sqrt_2Ck_sigmaG = pars["sqrt_2Ck_sigmaG"]
    X_j = pars["X_j"]

    temp = -alpha * s * sum_fail * sqrt_2Ck_sigmaG \
            + np.sum(exp_epsilon * (1 - np.exp(-X_j * s * sqrt_2Ck_sigmaG * alpha))) \
            - s ** 2
    
    return np.exp(temp)


def gauss_hermite_adaptive_integral(k, exp_epsilon, pars, m_points):
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
        temp = np.sum([w_weights[i] * gh_integrand_adaptive(x_points[i], pars, exp_epsilon) for i in range(0,m_points - 1)])
        temp += w_weights[-1]
    
    elif m_points == 5:

        x_points[np.arange(0,m_points-1,2)] = [2.0201828704561, 0.95857246461382]
        x_points[np.arange(1,m_points-1,2)] =  - x_points[np.arange(0,m_points-1,2)]

        w_weights[np.arange(0,m_points,2)] = [1.181488625536, 0.98658099675143, 0.94530872048294]
        w_weights[np.arange(1,m_points, 2)] = w_weights[np.arange(0, m_points - 1, 2)]

        x_points = pars["sigma"] * x_points
        temp = np.sum([w_weights[i] * gh_integrand_adaptive(x_points[i], pars, exp_epsilon) for i in range(0,m_points - 1)])
        temp += w_weights[-1]

    elif m_points == 7:

        x_points[np.arange(0,m_points-1,2)] = [2.6519613568352, 1.6735516287675, 0.81628788285897]
        x_points[np.arange(1,m_points-1,2)] =  - x_points[np.arange(0,m_points-1,2)]

        w_weights[np.arange(0,m_points,2)] = [1.1013307296103, 0.8971846002252, 0.8286873032836, 0.81026461755681]
        w_weights[np.arange(1,m_points, 2)] = w_weights[np.arange(0, m_points - 1, 2)]

        x_points = pars["sigma"] * x_points
        temp = np.sum([w_weights[i] * gh_integrand_adaptive(x_points[i], pars, exp_epsilon) for i in range(0,m_points - 1)])
        temp += w_weights[-1]
    

    else:
        raise ValueError('Possible number of quad points: 3, 5, 7 ')



    return pars["sigma"]*temp


def marginal_likelihood_vec_calc(pars, exp_epsilon, n):

    '''Calculate the marginal likelihood vectors for components not equal to zero
    Input:
    - pars: parameters dictionary
    - exp_epsilon: exponentiated residuals
    - n: number of quadrature points
    - marker: column of the markers matrix
    
    Ouput:
    - marginal_likelihoods: vector with probabilities for each mixture'''
    
    exp_sum = np.sum(exp_epsilon * pars["Xj_sqrt"])

    marginal_likelihood = [pars["marginal_likelihoods"][0],]

    for k in range(len(pars["Ck"])):
        pars["sigma"] = 1/np.sqrt(1 + pars["alpha"]**2 * pars["sigma_g"] * pars["Ck"][k] * exp_sum)
        temp = pars["pi_L"][k+1] \
            * gauss_hermite_adaptive_integral(k=k, exp_epsilon=exp_epsilon, pars=pars, m_points = n)

        # if np.isinf(temp):
        #     print(k, exp_sum, pars["pi_L"][k+1], gauss_hermite_adaptive_integral(k=k, exp_epsilon=exp_epsilon, pars=pars, m_points = n, marker=marker ))
       
        marginal_likelihood.append(temp)

    return np.array(marginal_likelihood)


    
if __name__ == "__main__":
    pass
    