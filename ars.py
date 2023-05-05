import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.colors import LinearSegmentedColormap
from cycler import cycler

import helpers 

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Palatino'], 'size':14})
rc('text', usetex=True)
plt.rc('axes', prop_cycle=(cycler('color', ['red', 'gray', 'black'])))#, 'blue', 'green'])))

def g_u(x, xs, hs, dhdxs):
    """Function to compute the envelope function. 
    Input:
    -x: x values
    -xs: abscissa
    -hs: h(xs)
    -dhdxs: h'(xs)
    
    First we compute the intersection values between the tangent functions.
    We approximate the function to the tangent line defined by the most proximate xs point.
    It has the form y = h'(xs) * (x-xs) + h(x)
    h = log f (being f the function we want to sample) so we exponentiate the line.
    """
    z, _ = compute_points_of_intersection_and_intercepts(xs, hs, dhdxs)
    i = np.searchsorted(z, x)
    
    return np.exp(dhdxs[i] * (x - xs[i]) + hs[i])

    

def g_l(x, xs, hs):
    """Compute the squeezing function
    Input:
    -x: x values
    -xs: abscissa
    -hs: h(xs) envelope function
    
    We want to find the minimum distant abscissa point to each x point. And calculate the line function.
    We are finding the line that joins the h(x) functions and the x, so that we get a slope and 
    then we return a line (exp). This is joining basically two abjacent abscissae lines.
    """
    
    if all(x < xs) or all(x > xs):
        return 0.
    
    else:
        i = np.searchsorted(xs, x)
        m = (hs[i] - hs[i-1]) / (xs[i] - xs[i-1])
        
        return np.exp(hs[i-1] + (x - xs[i-1]) * m)


def compute_points_of_intersection_and_intercepts(x, h, dhdx):
    '''
    Function to find the intersection point of two functions. 
    h(x) and tangent line at x, which is given by y=mx + c
    where m is the slope, given by the derivative at x dhdx(x).
    
    Input:
    -x: abscissae points
    -h: h(x) log concave function
    -dhdx: h'(x) derivate in points

    First we find the intercept c of the line h(x) = dhdx(x)*x + c.
    Now we have the tangent lines of the functions at the x points.
    We can find the intersection between this lines (which 
    is the envelope function gu).
    y=m1 x + c1 and y=m2 x + c2. We equal the y and obtain the intersection:
    x = c2 - c1/m1 - m2
    
    Output:
    -c: intercepts of the tangent lines
    -z: intersection points of the tangent lines'''
    
    c = h - dhdx * x
    z = (c[1:] - c[:-1]) / (dhdx[:-1] - dhdx[1:])
    
    return z, c

def log_gaussian(mean, variance):
    '''computes the log-likelihood and the derivative of the log gaussian function'''
    return lambda x : (- 0.5 * (x - mean) ** 2 / variance)
                       
def log_gaussian_dev(mean, variance):
    return lambda x: (- (x - mean) / variance)

def envelope_limits_and_unnormalised_probabilities(xs, hs, dhdxs, lims = (float('-inf'),float('inf'))):
    '''Determine the left and right limits and compute the unormalised probabilities
    of each piecewise exponential. 
    First we get the intersection points (z) which define the different regions.
    E.g. If we have a intersection point in x=1 then the domain will be divided in [-inf,0], [0,inf].
    The envelope is given by the line y=h'(x-x1) + c (this is the envelope function gu). 
    We compute the exponent of this function to get the probability values. 
    '''
        
    # Compute the points of intersection of the lines making up the envelope
    z, c = compute_points_of_intersection_and_intercepts(xs, hs, dhdxs)

    # Left-right endpoints for each piece in the piecewise envelope
    limits = np.concatenate([[lims[0]], z, [lims[1]]])
    limits = np.stack([limits[:-1], limits[1:]], axis=-1)
    probs = (np.exp(dhdxs * limits[:, 1]) - np.exp(dhdxs * limits[:, 0])) * np.exp(c)
    
    # Catch any intervals where dhdx was zero
    idx_nonzero = np.where(dhdxs != 0.)
    probs[idx_nonzero] = probs[idx_nonzero] / dhdxs[idx_nonzero]
    
    idx_zero = np.where(dhdxs == 0.)
    probs[idx_zero] = ((limits[:, 1] - limits[:, 0]) * np.exp(c))[idx_zero]
    
    return limits, probs


def sample_envelope(xs, hs, dhdxs, bounds):
    
    limits, all_probs = envelope_limits_and_unnormalised_probabilities(xs, hs, dhdxs)

    include = helpers.find_positions_between_limits(limits, bounds)
    probs = np.zeros(len(all_probs))
    probs[include] = all_probs[include]

    if len(include) == 0:
        print(limits)
    

    probs = probs/np.sum(probs)
    
    # Randomly chosen interval in which the sample lies
    i = np.random.choice(np.arange(probs.shape[0]), p=probs)

    # Sample u = Uniform(0, 1)
    u = np.random.uniform()
    
    # Invert i^th piecewise exponential CDF to get a sample from that interval
    if dhdxs[i] == 0.:
        return u * (limits[i, 1] - limits[i, 0]) + limits[i, 0]
        
    else:
        x = np.log(u * np.exp(dhdxs[i] * limits[i, 1]) \
                + (1 - u) * np.exp(dhdxs[i] * limits[i, 0]))
        x = x / dhdxs[i] 
    
        return x
    
def initialise_abcissa(x0, log_unnorm_prob, derivative, npoints, bounds):
    '''
    Function to initialize the abcissa. We first take values until we find a positive and negative derivative.
    Then we sample points from this interval, we can also sample between bounds, but it may happen that the ml 
    is way off, and then our sampler is not very precise.
    '''

    
    # Expand to the left/right until the abcissa is correctly initialised
    xs = np.array([x0], dtype='float')
    hs = log_unnorm_prob(xs)
    dhdxs = derivative(xs)
    dx = -1.
    
    while True:
        
        if dx < 0. and dhdxs[0] > 0.:
            dx = 1.

        elif dx > 0. and dhdxs[-1] < 0.:

            # for x in bounds:
            #     insert_idx = np.searchsorted(xs, x)
            #     h, dhdx = log_unnorm_prob(x), derivative(x)
            #     xs = np.insert(xs, insert_idx, x)
            #     hs = np.insert(hs, insert_idx, h)
            #     dhdxs = np.insert(dhdxs, insert_idx, dhdx)
            # i commented this because adding the bounds is not going to make them appear into the limits

            if len(xs) >= npoints:
                break

            else:

                for _ in range(npoints-len(xs)):
                    x = helpers.find_missing_number(xs, bounds)
                    h, dhdx = log_unnorm_prob(x), derivative(x)
                    insert_idx = np.searchsorted(xs, x)

                    xs = np.insert(xs, insert_idx, x)
                    hs = np.insert(hs, insert_idx, h)
                    dhdxs = np.insert(dhdxs, insert_idx, dhdx)
                
                break
        
        insert_idx = 0 if dx < 0 else len(xs)
        
        x = xs[0 if dx < 0 else -1] + dx
        
        h, dhdx = log_unnorm_prob(x), derivative(x)
        xs = np.insert(xs, insert_idx, x)
        hs = np.insert(hs, insert_idx, h)
        dhdxs = np.insert(dhdxs, insert_idx, dhdx)
        
        dx = dx * 2
        
    return xs, hs, dhdxs



def adaptive_rejection_sampling(x0, log_unnorm_prob, derivative, num_samples, ini_points=100, bounds = (float("-inf"), float("inf"))):
    
    xs, hs, dhdxs = initialise_abcissa(x0=x0, log_unnorm_prob = log_unnorm_prob, derivative=derivative, 
                                       npoints=ini_points, bounds=bounds)
    
    samples = []
    while len(samples) < num_samples:
        
        x = sample_envelope(xs, hs, dhdxs, bounds)
        
        gl = g_l(x, xs, hs)
        gu = g_u(x, xs, hs, dhdxs)

        
        u = np.random.rand()

        h, dhdx = log_unnorm_prob(x), derivative(x)
        # Squeezing test
        if u * gu <= gl:
            samples.append(x)

        # Rejection test
        elif u * gu <= np.exp(h):
            samples.append(x)

            i = np.searchsorted(xs, x)

            xs = np.insert(xs, i, x)
            hs = np.insert(hs, i, h)
            dhdxs = np.insert(dhdxs, i, dhdx)
        
    return samples, xs
    


def main(): 

    # The log unnormalised density to illustrate
    log_prob = log_gaussian(0., 1.)

    # Points in the abcissa set and corresponding log-probabilities and gradients
    xs = np.array([-1., 0.1, 1.5])
    hs, dhdxs = log_prob(xs)

    # Locations to plot the log unnorm. density and envelope/squeezing functions
    x_plot = np.linspace(-2, 2, 200)
    log_probs = [log_prob(x)[0] for x in x_plot]
    gu = [g_u(x, xs, hs, dhdxs) for x in x_plot]
    gl = [g_l(x, xs, hs) for x in x_plot]

    # Plot the log unnormalised density, the envelope and squeezing functions
    plt.figure(figsize=(6, 3))
    plt.scatter(xs, hs, color='k', zorder=3)
    plt.plot(x_plot, log_probs, color='black', label='$\log~f = h$')
    plt.plot(x_plot, np.log(gu), color='red', label='$\log~g_u$')

    # Handle the case of negatively infinite gl, for plotting presentation
    floored_log_gl = np.log(np.maximum(np.array(gl), np.ones_like(gl) * 1e-9))
    plt.plot(x_plot, floored_log_gl, color='green', label='$\log~g_l$')

    # Plot formatting
    plt.xlim([-2, 2])
    plt.ylim([-3, 1])
    plt.xlabel('$x$', fontsize=18)
    plt.ylabel('$\log~f(x)$', fontsize=18)
    plt.legend()
    plt.show(block=False)

    

    x_plot = np.linspace(-4., 4., 200)
    _, probs = envelope_limits_and_unnormalised_probabilities(xs, hs, dhdxs)

    samples = [sample_envelope(xs, hs, dhdxs) for i in range(10000)]
    gu = [g_u(x, xs, hs, dhdxs) / np.sum(probs) for x in x_plot]

    # Plot samples and envelope
    plt.figure(figsize=(6, 3))
    plt.plot(x_plot,
            gu,
            color='red',
            label='Normalised $g_u$')

    plt.hist(samples,
            density=True,
            bins=100,
            color='gray',
            alpha=0.5,
            label='Envelope samples')

    # Plot formatting
    plt.title('', fontsize=20)
    plt.xlim([-4, 4])
    plt.ylim([0, 0.5])
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('$x$', fontsize=18)
    plt.ylabel('$f(x)~/~Z$', fontsize=18)
    plt.legend()
    plt.show(block=False)

    np.random.seed(0)

    target_mean = 0.
    target_variance = 1.

    x0 = 1.
    num_samples = 10000

    log_unnorm_prob = log_gaussian(mean=target_mean, variance=target_variance)

    samples = adaptive_rejection_sampling(x0=x0, log_unnorm_prob=log_unnorm_prob, num_samples=num_samples)

    # Log probabilites for plotting the target
    x_plot = np.linspace(-4, 4, 200)
    log_probs = [np.exp(log_prob(x)[0]) / (2 * np.pi) ** 0.5 for x in x_plot]

    # Plot samples and target
    plt.figure(figsize=(6, 3))

    plt.hist(samples,
            density=True,
            bins=50,
            color='gray',
            alpha=0.5,
            label='Samples')
    plt.plot(x_plot,
            log_probs,
            color='black',
            label='Normalised $f(x)$')

    # Plot formatting
    plt.xlim([-4, 4])
    plt.ylim([0, 0.5])
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('$x$', fontsize=18)
    plt.ylabel('$f(x)~/~Z$', fontsize=18)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()