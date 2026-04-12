"""
Noncentral chi-square distribution
----------------------------------
"""

from mpmath import mp
from ..fun import marcumq, cmarcumq


__all__ = ['pdf', 'cdf', 'sf', 'support', 'mean', 'var']


@mp.extradps(5)
def pdf(x, k, lam):
    """
    PDF for the noncentral chi-square distribution.
    """
    if x < 0:
        return mp.zero
    x = mp.mpf(x)
    k = mp.mpf(k)
    lam = mp.mpf(lam)
    p = (mp.exp(-(x + lam)/2) * mp.power(x / lam, (k/2 - 1)/2) *
         mp.besseli(k/2 - 1, mp.sqrt(lam*x))/2)
    return p


@mp.extradps(5)
def cdf(x, k, lam):
    """
    CDF for the noncentral chi-square distribution.
    """
    if x <= 0:
        return mp.zero
    x = mp.mpf(x)
    k = mp.mpf(k)
    lam = mp.mpf(lam)
    c = cmarcumq(k/2, mp.sqrt(lam), mp.sqrt(x))
    return c


@mp.extradps(5)
def sf(x, k, lam):
    """
    Survival function for the noncentral chi-square distribution.
    """
    if x <= 0:
        return mp.one
    x = mp.mpf(x)
    k = mp.mpf(k)
    lam = mp.mpf(lam)
    s = marcumq(k/2, mp.sqrt(lam), mp.sqrt(x))
    return s


@mp.extradps(5)
def support(k, lam):
    """
    Support of the noncentral chi-square distribution.
    """
    k = mp.mpf(k)
    lam = mp.mpf(lam)
    return (mp.zero, mp.inf)


@mp.extradps(5)
def mean(k, lam):
    """
    Mean of the noncentral chi-square distribution.
    """
    k = mp.mpf(k)
    lam = mp.mpf(lam)
    return k + lam


@mp.extradps(5)
def var(k, lam):
    """
    Variance of the noncentral chi-square distribution.
    """
    k = mp.mpf(k)
    lam = mp.mpf(lam)
    return 2 * (k + 2 * lam)
