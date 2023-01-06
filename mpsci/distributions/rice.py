"""
Rice distribution
-----------------

Parameter names and formulas are from the wikipedia article:

    https://en.wikipedia.org/wiki/Rice_distribution

SciPy has a different parametrization::

    mpsci               SciPy
    -----------------   -----------------------------------------------------
    pdf(x, nu, sigma)   scipy.stats.rice.pdf(x, nu/sigma, loc=0, scale=sigma)

"""

from mpmath import mp
from ..fun import marcumq, cmarcumq


__all__ = ['pdf', 'cdf', 'sf']


def pdf(x, nu, sigma):
    """
    PDF for the Rice distribution.
    """
    if x <= 0:
        return mp.zero
    with mp.extradps(5):
        x = mp.mpf(x)
        nu = mp.mpf(nu)
        sigma = mp.mpf(sigma)
        sigma2 = sigma**2
        p = ((x / sigma2) * mp.exp(-(x**2 + nu**2)/(2*sigma2)) *
             mp.besseli(0, x*nu/sigma2))
    return p


def cdf(x, nu, sigma):
    """
    CDF for the Rice distribution.
    """
    if x <= 0:
        return mp.zero
    with mp.extradps(5):
        x = mp.mpf(x)
        nu = mp.mpf(nu)
        sigma = mp.mpf(sigma)
        c = cmarcumq(1, nu/sigma, x/sigma)
    return c


def sf(x, nu, sigma):
    """
    Survival function for the Rice distribution.
    """
    if x <= 0:
        return mp.one
    with mp.extradps(5):
        x = mp.mpf(x)
        nu = mp.mpf(nu)
        sigma = mp.mpf(sigma)
        s = marcumq(1, nu/sigma, x/sigma)
    return s
