"""
Functions used by the weibull_min and weibull_max distributions.
"""

from mpmath import mp


def _validate_params(k, loc, scale):
    if k <= 0:
        raise ValueError('k must be positive')
    if scale <= 0:
        raise ValueError('scale must be positive')
    k = mp.mpf(k)
    loc = mp.mpf(loc)
    scale = mp.mpf(scale)
    return k, loc, scale


def _mle_k_eqn1(k, x):
    # Use this when the scale was also estimated from the data.
    # Do not use it if the scale was a given fixed value.
    lnx = [mp.log(xi) for xi in x]
    xk = [xi**k for xi in x]
    xklnx = [xki * lnxi for (xki, lnxi) in zip(xk, lnx)]
    t1 = mp.fsum(xklnx) / mp.fsum(xk)
    t2 = mp.fsum(lnx) / len(x)
    return t1 - t2 - 1/k


def _mle_k_eqn2(k, x, scale):
    # Use this equation to solve for k when the scale is a given fixed value.
    n = len(x)
    lnx = [mp.log(xi) for xi in x]
    xk = [xi**k for xi in x]
    xklnx = [xi**k*mp.log(xi) for xi in x]
    m1 = mp.fsum(xk)/n
    m2 = mp.fsum(xklnx)/n
    lnscale = mp.log(scale)
    return 1/k - lnscale + mp.fsum(lnx)/n + (1/scale**k)*(lnscale*m1 - m2)
