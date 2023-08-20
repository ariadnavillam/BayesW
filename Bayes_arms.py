#!/usr/bin/env python
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : ariadnavillam 
# Created Date: 01/04/23
# version ='1.0'
# 
# ---------------------------------------------------------------------------
""" Functions for Adaptive Rejection Sampling""" 

import numpy as np 
import sys
import math

import helpers 

YCEIL = 50
XEPS = 1e-5
YEPS = 0.1
EYEPS = 0.001

class Point:
    def __init__(self, x, y, ey, cum, f, pl, pr):
        self.x = x
        self.y = y
        self.ey = ey
        self.cum = cum
        self.f = f
        self.pl = pl
        self.pr = pr
    

class Envelope:
    def __init__(self, cpoint, npoint, ymax, p, convex):
        self.cpoint = cpoint
        self.npoint = npoint
        self.ymax = ymax
        self.p = p
        self.convex = convex



class Metropolis:
    def __init__(self, on, xprev, yprev):
        self.on = on
        self.xprev = xprev
        self.yprev = yprev

class Funbag:
    def __init__(self, myfunc):
        self.myfunc = myfunc  # User-defined function evaluating log density at x

def arms(xinit, ninit, xl, xr, myfunc, convex, npoint, dometrop, xprev,
         nsamp, qcent, xcent, ncent, xsamp):
    
    pwork = Point(0,0,0,0,0,None,None)   # a working point, not yet incorporated in envelope
    msamp = 0             # the number of x-values currently sampled
    lpdf = Funbag(myfunc)       # to hold density function and its data
    metrop = Metropolis(dometrop, None, None) # to hold bits for metropolis step

    # check requested envelope centiles
    for i in range(ncent):
        if qcent[i] < 0.0 or qcent[i] > 100.0:
            # percentage requesting centile is out of range
            return 1005
    
    # incorporate density function and its data into FUNBAG lpdf
    lpdf.myfunc = myfunc

    # set up initial envelope
    ini_out = initial(xinit, ninit, xl, xr, npoint, lpdf, convex, metrop)
    
    if type(ini_out) != int:
        err, env = ini_out
    else:
        err = ini_out

    if err:
        return err
    
    print(env.p)

    # finish setting up metropolis struct (can only do this after setting up env)
    if metrop.on:
        if xprev < xl or xprev > xr:
            # previous markov chain iterate out of range
            return 1007
        metrop.xprev = xprev
        metrop.yprev = perfunc(lpdf, env, xprev)
    
    # now do adaptive rejection
    while msamp < nsamp:
        # sample a new point
        
        sample(env, pwork)

        # perform rejection (and perhaps metropolis) tests
        i = test(env, pwork, lpdf, metrop)

        if i == 1:
            # point accepted
            xsamp.append(pwork.x)
            msamp += 1

        elif i != 0:
            # envelope error - violation without metropolis
            return 2000

    # nsamp points now sampled
    # calculate requested envelope centiles
    for i in range(ncent):
        invert(qcent[i] / 100.0, env, pwork)
        xcent[i] = pwork.x

    # free space
    del env.p
    del env
    del metrop

    return 0

def initial(xinit, ninit, xl, xr, npoint, lpdf, convex, metrop):
    
    q = None

    if ninit < 3:
        # too few initial points
        return 1001

    mpoint = 2 * ninit + 1

    if npoint < mpoint:
        # too many initial points
        return 1002

    if xinit[0] <= xl or xinit[ninit - 1] >= xr:
        # initial points do not satisfy bounds
        return 1003

    for i in range(1, ninit):
        if xinit[i] <= xinit[i - 1]:
            # data not ordered
            return 1004

    if convex < 0.0:
        # negative convexity parameter
        return 1008

    # copy convexity parameter address to env

    env = Envelope(mpoint, npoint, 0, [None]*npoint, convex)
    

    # initialize current number of function evaluations
    


    for i in range(mpoint):
        env.p[i] = Point(0.0, 0, 0, 0, 0, None, None)

    q = env.p
    # left bound
    q[0].x = xl
    q[0].f = 0
    q[0].pl = None
    q[0].pr = q[1]

    k = 0
    for j in range(1, mpoint - 1):
        q[j].f = 0
        q[j].pl = q[j - 1]
        q[j].pr = q[j + 1]

        if j % 2:
            # point on log density
            q[j].x = xinit[k]
            q[j].y = perfunc(lpdf, env, q[j].x)
            q[j].f = 1
            k += 1

    # right bound
    q[mpoint - 1].x = xr
    q[mpoint - 1].f = 0
    q[mpoint - 1].pl = q[mpoint - 2]
    q[mpoint - 1].pr = None

    env.p = q
    

    for j in range(0, mpoint, 2):
        if meet(q[j], env, metrop):
            # envelope violation without metropolis
            raise ValueError("envelope violation without metropolis")

    # exponentiate and integrate envelope

    

    cumulate(env)

    # note number of POINTs currently in envelope
    env.cpoint = mpoint

    return 0, env

def sample(env, p):
    '''To sample from apiecewise exponential envelope'''
    prob = np.random.uniform()

    invert(prob, env, p)

    return 

def invert(prob, env, p):
    q = env.p[0]

    # Find the rightmost point in the envelope
    while q.pr is not None:
        q = q.pr

    # Find the exponential piece containing the point implied by the probability
    u = prob * q.cum
    while q.pl.cum > u:
        q = q.pl
    

    # Set the left and right points of p and calculate the proportion within this piece
    p.pl = q.pl
    p.pr = q
    p.f = 0
    p.cum = u
    prop = (u - q.pl.cum)/(q.cum - q.pl.cum)
    
    # Calculate the required x-value
    if np.abs(q.pl.x - q.x) < XEPS: ## equal
        # Interval is of zero length
        p.x = q.x
        p.y = q.y
        p.ey = q.ey

    else:
        xl = q.pl.x
        xr = q.x
        yl = q.pl.y
        yr = q.y
        eyl = q.pl.ey
        eyr = q.ey

        if math.isclose(yr, yl, abs_tol=YEPS):
            # Linear approximation was used in integration in the function cumulate
            if np.abs(eyr - eyl) > EYEPS * abs(eyr + eyl):
                p.x = xl + ((xr - xl) / (eyr - eyl)) * (-eyl + np.sqrt((1. - prop) * eyl * eyl + prop * eyr * eyr))
            else:
                p.x = xl + (xr - xl) * prop
            p.ey = ((p.x - xl) / (xr - xl)) * (eyr - eyl) + eyl
            p.y = logshift(p.ey, env.ymax)
        else:
            # The piece was integrated exactly in the function cumulate
            p.x = xl + ((xr - xl) / (yr - yl)) * (-yl + logshift((1. - prop) * eyl + prop * eyr, env.ymax))
            p.y = ((p.x - xl) / (xr - xl)) * (yr - yl) + yl
            p.ey = expshift(p.y, env.ymax)
    
        # Guard against imprecision yielding a point outside the interval
        if p.x < xl - XEPS or p.x > xr + XEPS:
            print("EXIT1")
            sys.exit(1)

    return 

    

def test(env, p, lpdf, metrop):
    u = np.random.uniform() * p.ey
    y = logshift(u, env.ymax)

    if not metrop.on and p.pl.pl is not None and p.pr.pr is not None:
        ql = p.pl if p.pl.f else p.pl.pl
        qr = p.pr if p.pr.f else p.pr.pr
        ysqueez = (qr.y * (p.x - ql.x) + ql.y * (qr.x - p.x)) / (qr.x - ql.x)
        if y <= ysqueez:
            return 1

    ynew = perfunc(lpdf, env, p.x)

    if not metrop.on or (metrop.on and y >= ynew):
        p.y = ynew
        p.ey = expshift(p.y, env.ymax)
        p.f = 1

        if update(env, p, lpdf, metrop) == -1:
            return -1

        if y >= ynew:
            return 0
        else:
            return 1

    yold = metrop.yprev
    ql = env.p[0]
    while ql.pl is not None:
        ql = ql.pl
    while ql.pr.x < metrop.xprev:
        ql = ql.pr
    qr = ql.pr

    w = (metrop.xprev - ql.x) / (qr.x - ql.x)
    zold = ql.y + w * (qr.y - ql.y)
    znew = p.y
    if yold < zold:
        zold = yold
    if ynew < znew:
        znew = ynew
    w = ynew - znew - yold + zold
    if w > 0.0:
        w = 0.0

    if w > -YCEIL:
        w = np.exp(w)
    else:
        w = 0.0
    u = np.random.uniform()
    if u > w:
        p.x = metrop.xprev
        p.y = metrop.yprev
        p.ey = expshift(p.y,env.ymax)
        p.f = 1
        p.pl = ql
        p.pr = qr
    else:
    # trial point accepted by metropolis, so update previous markov 
    # hain iterate 
        metrop.xprev = p.x
        metrop.yprev = ynew
  
    return 1


def update(env, p, lpdf, metrop):
    if not p.f or env.cpoint > env.npoint - 2:
        # y-value has not been evaluated or no room for further points
        # Ignore this point
        return 0
    
    # Copy working POINT p to a new POINT q
    q = env.p[env.cpoint] = Point(p.x, p.y, 0, 0, p.f, None, None)
    env.cpoint += 1

    # Allocate an unused POINT for a new intersection
    m = env.p[env.cpoint] = Point(0, 0, 0, 0, 0, None, None)
    env.cpoint += 1
    m.f = 0

    if p.pl.f and not p.pr.f:
        # Left end of piece is on log density; right end is not
        # Set up new intersection in interval between p.pl and p
        m.pl = p.pl
        m.pr = q
        q.pl = m
        q.pr = p.pr
        m.pl.pr = m
        q.pr.pl = q
    elif not p.pl.f and p.pr.f:
        # Left end of interval is not on log density; right end is
        # Set up new intersection in interval between p and p.pr
        m.pr = p.pr
        m.pl = q
        q.pr = m
        q.pl = p.pl
        m.pr.pl = m
        q.pl.pr = q
    else:
        # This should be impossible
        raise ValueError("Error in intersection point")

    # Adjust position of q within interval if too close to an endpoint
    if q.pl.pl is not None:
        ql = q.pl.pl
    else:
        ql = q.pl
    if q.pr.pr is not None:
        qr = q.pr.pr
    else:
        qr = q.pr
    if q.x < (1. - XEPS) * ql.x + XEPS * qr.x:
        # q too close to left end of interval
        q.x = (1. - XEPS) * ql.x + XEPS * qr.x
        q.y = perfunc(lpdf, env, q.x)
    elif q.x > XEPS * ql.x + (1. - XEPS) * qr.x:
        # q too close to right end of interval
        q.x = XEPS * ql.x + (1. - XEPS) * qr.x
        q.y = perfunc(lpdf, env, q.x)

    # Revise intersection points
    if meet(q.pl, env, metrop):
        # Envelope violation without metropolis
        return 1

    if meet(q.pr, env, metrop):
        # Envelope violation without metropolis
        return 1

    if q.pl.pl is not None:
        if meet(q.pl.pl.pl, env, metrop):
            # Envelope violation without metropolis
            return 1

    if q.pr.pr is not None:
        if meet(q.pr.pr.pr, env, metrop):
            # Envelope violation without metropolis
            return 1

    # Exponentiate and integrate new envelope
    cumulate(env)

    return 0

def cumulate(env):
    qlmost = env.p[0]

    # Find left end of envelope

    while qlmost.pl is not None:
        qlmost = env.p[i]
        i+=1

    # Find maximum y-value: search envelope
    env.ymax = qlmost.y
    q = qlmost.pr

    while q is not None:
        if q.y > env.ymax:
            env.ymax = q.y
        q = q.pr

    # Exponentiate envelope
    q = qlmost
    while q is not None:
        q.ey = expshift(q.y, env.ymax)
        q = q.pr

    # Integrate exponentiated envelope
    qlmost.cum = 0.0
    q = qlmost.pr
    while q is not None:
        q.cum = q.pl.cum + area(q)
        q = q.pr

    return


def meet(q, env, metrop):
    
    if q.f:
        # This is not an intersection point
        raise ValueError("Not an intersection point")


    if q.pl != None and q.pl.pl.pl != None:
        # Chord gradient can be calculated at left end of interval
        gl = (q.pl.y - q.pl.pl.pl.y) / (q.pl.x - q.pl.pl.pl.x)
        il = 1
    else:
        il = 0

    if q.pr != None and q.pr.pr.pr != None:
        # Chord gradient can be calculated at right end of interval
        gr = (q.pr.y - q.pr.pr.pr.y) / (q.pr.x - q.pr.pr.pr.x)
        ir = 1
    else:
        ir = 0

    if q.pl != None and q.pr != None:
        # Chord gradient can be calculated across interval
        grl = (q.pr.y - q.pl.y) / (q.pr.x - q.pl.x)
        irl = 1
    else:
        irl = 0

    if irl and il and (gl < grl):
        # Convexity on left exceeds current threshold
        if not metrop.on:
            # Envelope violation without metropolis
            return 1
        # Adjust left gradient
        gl = gl + (1.0 + env.convex) * (grl - gl)

    if irl and ir and (gr > grl):
        # Convexity on right exceeds current threshold
        if not metrop.on:
            # Envelope violation without metropolis
            return 1
        # Adjust right gradient
        gr = gr + (1.0 + env.convex) * (grl - gr)

    if il and irl:
        dr = (gl - grl) * (q.pr.x - q.pl.x)
        if dr < YEPS:
            # Adjust dr to avoid numerical problems
            dr = YEPS

    if ir and irl:
        dl = (grl - gr) * (q.pr.x - q.pl.x)
        if dl < YEPS:
            # Adjust dl to avoid numerical problems
            dl = YEPS

    if il and ir and irl:
        # Gradients on both sides
        q.x = (dl * q.pr.x + dr * q.pl.x) / (dl + dr)
        q.y = (dl * q.pr.y + dr * q.pl.y + dl * dr) / (dl + dr)

    elif il and irl:
        # Gradient only on left side, but not right-hand bound
        q.x = q.pr.x
        q.y = q.pr.y + dr
    elif ir and irl:
        # Gradient only on right side, but not left-hand bound
        q.x = q.pl.x
        q.y = q.pl.y + dl
    elif il:
        # Right-hand bound
        q.y = q.pl.y + gl * (q.x - q.pl.x)
    elif ir:
        # Left-hand bound
        q.y = q.pr.y - gr * (q.pr.x - q.x)
    else:
        # Gradient on neither side - should be impossible
        print("EXIT4")
        sys.exit(31)

    if (q.pl != None and q.x < q.pl.x - XEPS) or (q.pr != None and q.x > q.pr.x + XEPS):
        # Intersection point outside interval (through imprecision)

        print(q.x, q.pl.x)
        print("EXIT5")
        sys.exit(32)


               
def area(q):
    '''find the area under a piece of exponentiated envelope (between a point and the point to the left)'''
    if q.pl is None:
        # This is the leftmost point in the envelope
        print("EXIT6")
        sys.exit(1)

    elif np.abs(q.pl.x - q.x) < XEPS: ## equal
        # Interval has zero length
        a = 0.0
    elif abs(q.y - q.pl.y) < YEPS:
        # Integrate straight line piece
        a = 0.5 * (q.ey + q.pl.ey) * (q.x - q.pl.x)
    else:
        # Integrate exponential piece
        a = ((q.ey - q.pl.ey) / (q.y - q.pl.y)) * (q.x - q.pl.x)
    return a
        
def logshift(y, y0):
    '''inverse of function expshift'''
    return np.log(y) + y0 - YCEIL

def expshift(y, y0):
    '''exponentiate shifted y without undeflow'''
    if y - y0 > -2 * YCEIL:
        return np.exp(y - y0 + YCEIL)
    else:
        return 0

def perfunc(lpdf, env, x):
    y = lpdf.myfunc(x)  # Evaluate density function

    return y


