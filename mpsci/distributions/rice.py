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

import mpmath
from ..fun import marcumq, cmarcumq


__all__ = ['pdf', 'cdf', 'sf']


def pdf(x, nu, sigma):
    """
    PDF for the Rice distribution.
    """
    x = mpmath.mpf(x)
    nu = mpmath.mpf(nu)
    sigma = mpmath.mpf(sigma)
    sigma2 = sigma**2
    p = ((x / sigma2) * mpmath.exp(-(x**2 + nu**2)/(2*sigma2)) *
         mpmath.besseli(0, x*nu/sigma2))
    return p


def cdf(x, nu, sigma):
    """
    CDF for the Rice distribution.
    """
    x = mpmath.mpf(x)
    nu = mpmath.mpf(nu)
    sigma = mpmath.mpf(sigma)
    c = cmarcumq(1, nu/sigma, x/sigma)
    return c


def sf(x, nu, sigma):
    """
    Survival function for the Rice distribution.
    """
    x = mpmath.mpf(x)
    nu = mpmath.mpf(nu)
    sigma = mpmath.mpf(sigma)
    s = marcumq(1, nu/sigma, x/sigma)
    return s
