"""
Truncated standard normal distribution
--------------------------------------

The implementations here are somewhat naive.  To check the results,
run multiple times with increasing mpmath precision.
"""

import mpmath
from . import normal


__all__ = ['pdf', 'logpdf', 'cdf', 'sf', 'invcdf', 'invsf', 'mean', 'var',
           'median']


def _validate_params(a, b):
    if a >= b:
        raise ValueError("'a' must be less than 'b'")


def _norm_delta_cdf(a, b):
    """
    Compute CDF(b) - CDF(a) for the standard normal distribution CDF.

    The function assumes a <= b.
    """
    with mpmath.extradps(5):
        if a == b:
            return mpmath.mp.zero
        if a > 0:
            delta = mpmath.ncdf(-a) - mpmath.ncdf(-b)
        else:
            delta = mpmath.ncdf(b) - mpmath.ncdf(a)
        return delta


def pdf(x, a, b):
    """
    PDF of the truncated standard normal distribution.
    """
    _validate_params(a, b)
    if x < a or x > b:
        return mpmath.mp.zero
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        a = mpmath.mpf(a)
        b = mpmath.mpf(b)
        delta = _norm_delta_cdf(a, b)
        return mpmath.npdf(x) / delta


def logpdf(x, a, b):
    """
    Natural logarithm of the PDF of the truncated standard normal distribution.
    """
    _validate_params(a, b)
    if x < a or x > b:
        return -mpmath.inf
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        a = mpmath.mpf(a)
        b = mpmath.mpf(b)
        delta = _norm_delta_cdf(a, b)
        return normal.logpdf(x) - mpmath.log(delta)


def cdf(x, a, b):
    """
    CDF of the truncated standard normal distribution.
    """
    _validate_params(a, b)
    if x <= a:
        return mpmath.mp.zero
    if x >= b:
        return mpmath.mp.one
    with mpmath.extradps(5):
        return _norm_delta_cdf(a, x) / _norm_delta_cdf(a, b)


def sf(x, a, b):
    """
    Survival function of the truncated standard normal distribution.
    """
    _validate_params(a, b)
    if x <= a:
        return mpmath.mp.one
    if x >= b:
        return mpmath.mp.zero
    with mpmath.extradps(5):
        return _norm_delta_cdf(x, b) / _norm_delta_cdf(a, b)


def invcdf(p, a, b):
    """
    Inverse of the CDF of the truncated standard normal distribution.

    This function is also known as the quantile function or the percent
    point function.
    """
    _validate_params(a, b)
    if p < 0 or p > 1:
        return mpmath.nan

    with mpmath.extradps(5):
        p = mpmath.mpf(p)
        a = mpmath.mpf(a)
        b = mpmath.mpf(b)

        p2 = p * _norm_delta_cdf(a, b) + mpmath.ncdf(a)
        x = normal.invcdf(p2)

    return x


def invsf(p, a, b):
    """
    Inverse of the survival function of the standard normal distribution.
    """
    _validate_params(a, b)
    if p < 0 or p > 1:
        return mpmath.nan

    with mpmath.extradps(5):
        p = mpmath.mpf(p)
        a = mpmath.mpf(a)
        b = mpmath.mpf(b)

        p2 = -p * _norm_delta_cdf(a, b) + mpmath.ncdf(b)
        x = normal.invcdf(p2)

    return x


def mean(a, b):
    """
    Mean of the truncated standard normal distribution.
    """
    _validate_params(a, b)
    with mpmath.extradps(5):
        return pdf(a, a, b) - pdf(b, a, b)


def var(a, b):
    """
    Variance of the truncated standard normal distribution.
    """
    _validate_params(a, b)
    with mpmath.extradps(5):
        a = mpmath.mpf(a)
        b = mpmath.mpf(b)
        pa = pdf(a, a, b)
        pb = pdf(b, a, b)
        # Avoid the possibility of inf*0:
        ta = 0 if pa == 0 else a*pa
        tb = 0 if pb == 0 else b*pb
        return 1 + ta - tb - (pa - pb)**2


def median(a, b):
    """
    Median of the truncated standard normal distribution.
    """
    _validate_params(a, b)
    with mpmath.extradps(5):
        return normal.invcdf((normal.cdf(a) + normal.cdf(b))/2)
