"""
Normal distribution
-------------------
"""

import mpmath
from ._common import _validate_p


__all__ = ['pdf', 'logpdf', 'cdf', 'sf', 'invcdf', 'mle']


def pdf(x, mu=0, sigma=1):
    """
    Normal distribution probability density function.
    """
    # Defined here for consistency, but this is just mpmath.npdf
    return mpmath.npdf(x, mu, sigma)


def logpdf(x, mu=0, sigma=1):
    """
    Logarithm of the PDF of the normal distribution.
    """
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        mu = mpmath.mpf(mu)
        sigma = mpmath.mpf(sigma)
        logp = (-mpmath.log(2*mpmath.pi)/2 - mpmath.log(sigma)
                - (x - mu)**2/(2*sigma**2))
    return logp


def cdf(x, mu=0, sigma=1):
    """
    Normal distribution cumulative distribution function.
    """
    # Defined here for consistency, but this is just mpmath.ncdf
    return mpmath.ncdf(x, mu, sigma)


def sf(x, mu=0, sigma=1):
    """
    Normal distribution survival function.
    """
    return mpmath.ncdf(-x + 2*mu, mu, sigma)


def invcdf(p, mu=0, sigma=1):
    """
    Normal distribution inverse CDF.

    This function is also known as the quantile function or the percent
    point function.
    """
    with mpmath.extradps(mpmath.mp.dps):
        p = _validate_p(p)
        mu = mpmath.mpf(mu)
        sigma = mpmath.mpf(sigma)

        a = mpmath.erfinv(2*p - 1)
        x = mpmath.sqrt(2)*sigma*a + mu
        return x


def invsf(p, mu=0, sigma=1):
    """
    Inverse of the survival function of the normal distribution.
    """
    with mpmath.extradps(mpmath.mp.dps):
        p = _validate_p(p)
        mu = mpmath.mpf(mu)
        sigma = mpmath.mpf(sigma)

        a = mpmath.erfinv(1 - 2*p)
        x = mpmath.sqrt(2)*sigma*a + mu
        return x


# XXX Add standard errors and confidence intervals for the fitted parameters.

def mle(x):
    """
    Normal distribution maximum likelihood parameter estimation.

    Returns (mu, sigma).
    """
    x = [mpmath.mpf(t) for t in x]
    N = len(x)
    meanx = sum(x) / N
    var = sum((xi - meanx)**2 for xi in x) / N
    sigma = mpmath.sqrt(var)
    return meanx, sigma
