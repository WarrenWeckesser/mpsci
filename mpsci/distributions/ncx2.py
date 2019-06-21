"""
Noncentral chi-square distribution
----------------------------------
"""

import mpmath
from ..fun import marcumq, cmarcumq


__all__ = ['pdf', 'cdf', 'sf', 'mean', 'variance']


def pdf(x, k, lam):
    """
    PDF for the noncentral chi-square distribution.
    """
    x = mpmath.mpf(x)
    k = mpmath.mpf(k)
    lam = mpmath.mpf(lam)
    p = (mpmath.exp(-(x + lam)/2) * mpmath.power(x / lam, (k/2 - 1)/2) *
         mpmath.besseli(k/2 - 1, mpmath.sqrt(lam*x))/2)
    return p


def cdf(x, k, lam):
    """
    CDF for the noncentral chi-square distribution.
    """
    x = mpmath.mpf(x)
    k = mpmath.mpf(k)
    lam = mpmath.mpf(lam)
    c = cmarcumq(k/2, mpmath.sqrt(lam), mpmath.sqrt(x))
    return c


def sf(x, k, lam):
    """
    Survival function for the noncentral chi-square distribution.
    """
    x = mpmath.mpf(x)
    k = mpmath.mpf(k)
    lam = mpmath.mpf(lam)
    s = marcumq(k/2, mpmath.sqrt(lam), mpmath.sqrt(x))
    return s


def mean(k, lam):
    """
    Mean of the noncentral chi-square distribution.
    """
    k = mpmath.mpf(k)
    lam = mpmath.mpf(lam)
    return k + lam


def variance(k, lam):
    """
    Variance of the noncentral chi-square distribution.
    """
    k = mpmath.mpf(k)
    lam = mpmath.mpf(lam)
    return 2*(k + 2*lam)
