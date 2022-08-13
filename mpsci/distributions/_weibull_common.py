"""
Functions used by the weibull_min and weibull_max distributions.
"""

import mpmath


def _validate_params(k, loc, scale):
    if k <= 0:
        raise ValueError('k must be positive')
    if scale <= 0:
        raise ValueError('scale must be positive')
    k = mpmath.mp.mpf(k)
    loc = mpmath.mp.mpf(loc)
    scale = mpmath.mp.mpf(scale)
    return k, loc, scale


def _validate_p(p):
    if p < 0 or p > 1:
        raise ValueError('p must be in the interval [0, 1]')
    return mpmath.mp.mpf(p)


def _mle_k_eqn1(k, x):
    # Use this when the scale was also estimated from the data.
    # Do not use it if the scale was a given fixed value.
    lnx = [mpmath.log(xi) for xi in x]
    xk = [xi**k for xi in x]
    xklnx = [xki * lnxi for (xki, lnxi) in zip(xk, lnx)]
    t1 = mpmath.fsum(xklnx) / mpmath.fsum(xk)
    t2 = mpmath.fsum(lnx) / len(x)
    return t1 - t2 - 1/k


def _mle_k_eqn2(k, x, scale):
    # Use this equation to solve for k when the scale is a given fixed value.
    n = len(x)
    lnx = [mpmath.log(xi) for xi in x]
    xk = [xi**k for xi in x]
    xklnx = [xi**k*mpmath.log(xi) for xi in x]
    m1 = mpmath.fsum(xk)/n
    m2 = mpmath.fsum(xklnx)/n
    lnscale = mpmath.log(scale)
    return 1/k - lnscale + mpmath.fsum(lnx)/n + (1/scale**k)*(lnscale*m1 - m2)


def _median(x):
    """
    Compute the median of the sequence x.
    """
    xs = sorted(x)
    n = len(xs)
    m = n // 2
    if n & 1:
        med = mpmath.mpf(xs[m])
    else:
        med = mpmath.fsum(xs[m - 1:m + 1])/2
    return med
