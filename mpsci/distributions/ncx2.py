"""
Noncentral chi-square distribution
----------------------------------
"""

from mpmath import mp
from ..fun import marcumq, cmarcumq


__all__ = ['pdf', 'cdf', 'sf', 'mean', 'variance']


def pdf(x, k, lam):
    """
    PDF for the noncentral chi-square distribution.
    """
    if x < 0:
        return mp.zero
    with mp.extradps(5):
        x = mp.mpf(x)
        k = mp.mpf(k)
        lam = mp.mpf(lam)
        p = (mp.exp(-(x + lam)/2) * mp.power(x / lam, (k/2 - 1)/2) *
             mp.besseli(k/2 - 1, mp.sqrt(lam*x))/2)
    return p


def cdf(x, k, lam):
    """
    CDF for the noncentral chi-square distribution.
    """
    if x <= 0:
        return mp.zero
    with mp.extradps(5):
        x = mp.mpf(x)
        k = mp.mpf(k)
        lam = mp.mpf(lam)
        c = cmarcumq(k/2, mp.sqrt(lam), mp.sqrt(x))
    return c


def sf(x, k, lam):
    """
    Survival function for the noncentral chi-square distribution.
    """
    if x <= 0:
        return mp.one
    with mp.extradps(5):
        x = mp.mpf(x)
        k = mp.mpf(k)
        lam = mp.mpf(lam)
        s = marcumq(k/2, mp.sqrt(lam), mp.sqrt(x))
    return s


def mean(k, lam):
    """
    Mean of the noncentral chi-square distribution.
    """
    k = mp.mpf(k)
    lam = mp.mpf(lam)
    return k + lam


def variance(k, lam):
    """
    Variance of the noncentral chi-square distribution.
    """
    k = mp.mpf(k)
    lam = mp.mpf(lam)
    return 2*(k + 2*lam)
