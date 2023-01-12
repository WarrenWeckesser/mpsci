"""
Normal distribution
-------------------
"""

from mpmath import mp
from ._common import _validate_p


__all__ = ['pdf', 'logpdf', 'cdf', 'sf', 'invcdf', 'mle']


def pdf(x, mu=0, sigma=1):
    """
    Normal distribution probability density function.
    """
    # Defined here for consistency, but this is just mp.npdf
    return mp.npdf(x, mu, sigma)


def logpdf(x, mu=0, sigma=1):
    """
    Logarithm of the PDF of the normal distribution.
    """
    with mp.extradps(5):
        x = mp.mpf(x)
        mu = mp.mpf(mu)
        sigma = mp.mpf(sigma)
        logp = (-mp.log(2*mp.pi)/2 - mp.log(sigma)
                - (x - mu)**2/(2*sigma**2))
    return logp


def cdf(x, mu=0, sigma=1):
    """
    Normal distribution cumulative distribution function.
    """
    # Defined here for consistency, but this is just mp.ncdf
    return mp.ncdf(x, mu, sigma)


def sf(x, mu=0, sigma=1):
    """
    Normal distribution survival function.
    """
    with mp.extradps(5):
        x = mp.mpf(x)
        mu = mp.mpf(mu)
        sigma = mp.mpf(sigma)
        return mp.ncdf(-x + 2*mu, mu, sigma)


def invcdf(p, mu=0, sigma=1):
    """
    Normal distribution inverse CDF.

    This function is also known as the quantile function or the percent
    point function.
    """
    with mp.extradps(mp.dps):
        p = _validate_p(p)
        mu = mp.mpf(mu)
        sigma = mp.mpf(sigma)

        a = mp.erfinv(2*p - 1)
        x = mp.sqrt(2)*sigma*a + mu
        return x


def invsf(p, mu=0, sigma=1):
    """
    Inverse of the survival function of the normal distribution.
    """
    with mp.extradps(mp.dps):
        p = _validate_p(p)
        mu = mp.mpf(mu)
        sigma = mp.mpf(sigma)

        a = mp.erfinv(1 - 2*p)
        x = mp.sqrt(2)*sigma*a + mu
        return x


# XXX Add standard errors and confidence intervals for the fitted parameters.

def mle(x):
    """
    Normal distribution maximum likelihood parameter estimation.

    Returns (mu, sigma).
    """
    x = [mp.mpf(t) for t in x]
    N = len(x)
    meanx = sum(x) / N
    var = sum((xi - meanx)**2 for xi in x) / N
    sigma = mp.sqrt(var)
    return meanx, sigma
