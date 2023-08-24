#!/usr/bin/env python
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : ariadnavillam 
# Created Date: 01/04/23
# version ='1.0'
# Based on the code from: https://random-walks.org/content/misc/ars/ars.html#
# 
# ---------------------------------------------------------------------------
""" Functions for Adaptive Rejection Sampling""" 

import numpy as np 
import helpers 
import matplotlib.pyplot as plt

YCEIL = 50
XEPS = 1e-5
YEPS = 0.1
EYEPS = 0.001

def g_u(x, xs, hs, dhdxs, ymax):
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
    
    return expshift(dhdxs[i] * (x - xs[i]) + hs[i], ymax)

    

def g_l(x, xs, hs, ymax):
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
        return expshift(hs[i-1] + (x - xs[i-1]) * m, ymax)


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
    '''computes the log-likelihood of the gaussian distribution'''
    return lambda x : (- 0.5 * (x - mean) ** 2 / variance)
                       
def log_gaussian_dev(mean, variance):
    '''computes the derivative of the log-likelihood  of a gaussian distribution'''
    return lambda x: (- (x - mean) / variance)

def envelope_limits_and_unnormalised_probabilities(xs, hs, dhdxs, ymax):
    '''
    Determine the left and right limits and compute the unormalised probabilities
    of each piecewise exponential. 
    First we get the intersection points (z) which define the different regions.
    E.g. If we have a intersection point in x=1 then the domain will be divided in [-inf,0], [0,inf].
    The envelope is given by the line y=h'(x-x1) + c (this is the envelope function gu). 
    We compute the exponent of this function to get the probability values. 

    Input:
    - xs: abscissa points
    - hs: f(x)
    - dhdxs: f'(x)
    - lims: limits of the division (if this is changed it usually gives problems)

    Output:
    - limits: array of limits of x space. we compute this by dividing the domain 
    using the intersection points.
    - probs: probability of each region defined by the limits. 
    '''

    lims = (min(xs), max(xs))
        
    # Compute the points of intersection of the lines making up the envelope
    z, c = compute_points_of_intersection_and_intercepts(xs, hs, dhdxs)

    # Left-right endpoints for each piece in the piecewise envelope
    limits = np.concatenate([[lims[0]], z, [lims[1]]])
    limits = np.stack([limits[:-1], limits[1:]], axis=-1)

    y_arr_1 = dhdxs * limits[:, 1] + c
    y_arr_0 = dhdxs * limits[:, 0] + c

    probs = np.array([expshift(y, ymax) for y in y_arr_1]) - np.array([expshift(y, ymax) for y in y_arr_0])

    # Catch any intervals where dhdx was zero
    idx_nonzero = np.where(dhdxs != 0.)[0]
    probs[idx_nonzero] = probs[idx_nonzero] / dhdxs[idx_nonzero]
    
    idx_zero = np.where(dhdxs == 0.)[0]
    if len(idx_zero) != 0:
        
        probs[idx_zero] = ((limits[:, 1] - limits[:, 0]) * np.exp(c))[idx_zero]

    
    return limits, probs


def sample_envelope(xs, hs, dhdxs, xl, xr, ymax):
    # this part of the code is the one that gives more problems
    limits, probs = envelope_limits_and_unnormalised_probabilities(xs, hs, dhdxs, ymax)
    
    
    if np.any(probs < 0):
        print(xl, xr, probs[probs < 0], xs[probs < 0])

    probs = probs/np.sum(probs)
    
    # Randomly chosen interval in which the sample lies
    i = np.random.choice(np.arange(probs.shape[0]), p=probs)

    # Sample u = Uniform(0, 1)
    u = np.random.uniform()
    
    # Invert i^th piecewise exponential CDF to get a sample from that interval
    if dhdxs[i] == 0.:
        return u * (limits[i, 1] - limits[i, 0]) + limits[i, 0]
        
    else:
        x = np.logaddexp(np.log(u) + dhdxs[i] * limits[i, 1], np.log(1 - u) + dhdxs[i] * limits[i, 0])
        x = x / dhdxs[i] 
    
        return x
    
def initialise_abcissa(xinit, ninit, log_unnorm_prob, derivative, npoints, xl, xr):
    '''
    Function to initialize the abcissa. We first take values until we find a positive and negative derivative.
    Then we sample points from this interval, we can also sample between bounds, but it may happen that the ml 
    is way off, and then our sampler is not very precise.

    Input:
    - x0: initial value (our estimate of the parameter)
    - log_unnorm_prob: log probability function
    - derivative: derivative of the log prob
    - npoints: number of points we want to initialise our abscissa
    - bounds: limits of the points

    Ouput:
    -xs: abscissa points
    -hs: f(xs)
    -dhdxs: f'(xs)

    '''

    if ninit < 3:
        # too few initial points
        raise ValueError("xinit len")


    if xinit[0] <= xl or xinit[ninit - 1] >= xr:
        # initial points do not satisfy bounds
        raise ValueError("xinit out of limits")

    
    for i in range(1, ninit):
        if xinit[i] <= xinit[i - 1]:
            # data not ordered
            raise ValueError("xinit not order")
    
    # Expand to the left/right until the abcissa is correctly initialised
    xs = xinit
    hs = np.array([log_unnorm_prob(x) for x in xs])
    dhdxs = np.array([derivative(x) for x in xs])
    
    if (dhdxs[0] > 0.) and dhdxs[-1] <0.:  
        pass
    else:
        xs, hs, dhdxs = reinitilize_abscissa(xs, hs, dhdxs, log_unnorm_prob, derivative, xl, xr)
        plt.plot(xs, hs)
        plt.show()
        

    points = int((npoints - ninit)/(ninit-1))
    x_new = []
    for i in range(ninit - 1):
        x_new.append(np.linspace(xinit[i], xinit[i+1], points + 2)[1:-1])

    xs = np.sort(np.union1d(np.array(x_new).flatten(), xs))

    hs = np.array([log_unnorm_prob(x) for x in xs])
    dhdxs = np.array([derivative(x) for x in xs])

    # while True:
        
    #     if dx < 0. and dhdxs[0] > 0.:
    #         # when we have found a positive derivative to the left now we look to the right
    #         dx = 1. * ddx

    #     elif dx > 0. and dhdxs[-1] < 0.:
    #         #when we find a negative derivative to the right we are finished


    #         # for x in bounds:
    #         #     insert_idx = np.searchsorted(xs, x)
    #         #     h, dhdx = log_unnorm_prob(x), derivative(x)
    #         #     xs = np.insert(xs, insert_idx, x)
    #         #     hs = np.insert(hs, insert_idx, h)
    #         #     dhdxs = np.insert(dhdxs, insert_idx, dhdx)
    #         # i commented this because adding the bounds is not going to make them appear into the limits

    #         if len(xs) >= npoints:
    #             break

    #         else:
    #             # now we add points to have a better envelope function
    #             
        
        # insert_idx = 0 if dx < 0 else len(xs)
        
        # x = xs[0 if dx < 0 else -1] + dx
        
        # h, dhdx = log_unnorm_prob(x), derivative(x)

        # xs = np.insert(xs, insert_idx, x)
        # hs = np.insert(hs, insert_idx, h)
        # dhdxs = np.insert(dhdxs, insert_idx, dhdx)
        
        # dx = dx * 2
        
    return xs, hs, dhdxs

def reinitilize_abscissa(xs, hs, dhdxs, log_unnorm_prob, derivative, xl, xr):
    
    
    dx = -0.1
    counter = 0
    while True:
        counter +=1
        if counter > 100:
            raise ValueError("Could not find abcissa")
        
        if dx < 0. and dhdxs[0] > 0.:
            dx = 1.
            
        elif dx > 0. and dhdxs[-1] < 0.:
            break
        
        insert_idx = 0 if dx < 0 else len(xs)
        
        x = xs[0 if dx < 0 else -1] + dx
        
        if x < xl and dx < 0.:
            if derivative(xl) > 0.:
                x = xl
            else:
                print("ARS: Init points outside of left limit.")
                
        elif x > xr and dx > 0.:
            if derivative(xr) < 0.:
                x = xr
            else:
                print("ARS: Init points outside of right limit.")
            
        h = log_unnorm_prob(x)
        dhdx = derivative(x)

        xs = np.insert(xs, insert_idx, x)
        hs = np.insert(hs, insert_idx, h)
        dhdxs = np.insert(dhdxs, insert_idx, dhdx)
        
        dx = dx * 2
        
    return xs, hs, dhdxs

def ars(xinit, ninit, xl, xr, log_unnorm_prob, derivative, npoints,
                                nsamp):
    
    xs, hs, dhdxs = initialise_abcissa(xinit, ninit, log_unnorm_prob, derivative, npoints, xl, xr)
    
    xsamp = []
    while len(xsamp) < nsamp:
        ymax = hs.max()
        
        x = sample_envelope(xs, hs, dhdxs, xl, xr, ymax)
        
        gl = g_l(x, xs, hs, ymax)
        gu = g_u(x, xs, hs, dhdxs, ymax)

        
        u = np.random.rand()

        h, dhdx = log_unnorm_prob(x), derivative(x)
        # Squeezing test
        if u * gu <= gl:
            xsamp.append(x)

        # Rejection test
        elif u * gu <= expshift(h, ymax):
            xsamp.append(x)

            i = np.searchsorted(xs, x)

            xs = np.insert(xs, i, x)
            hs = np.insert(hs, i, h)
            dhdxs = np.insert(dhdxs, i, dhdx)
        
    return xsamp
    
def logshift(y, y0):
    '''inverse of function expshift'''
    return np.log(y) + y0 - YCEIL

def expshift(y, y0):
    '''exponentiate shifted y without undeflow'''
    if y - y0 > -2 * YCEIL:
        return np.exp(y - y0 + YCEIL)
    else:
        return 0



if __name__ == "__main__":
    pass