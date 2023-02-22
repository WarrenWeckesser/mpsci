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


__all__ = ['pdf', 'logpdf', 'cdf', 'sf', 'mean']


def _validate_params(nu, sigma):
    if nu < 0:
        raise ValueError('nu must be nonnegative')
    if sigma <= 0:
        raise ValueError('sigma must be greater than 0')
    return mp.mpf(nu), mp.mpf(sigma)


def pdf(x, nu, sigma):
    """
    PDF for the Rice distribution.
    """
    with mp.extradps(5):
        nu, sigma = _validate_params(nu, sigma)
        x = mp.mpf(x)
        if x <= 0:
            return mp.zero
        sigma2 = sigma**2
        p = ((x / sigma2) * mp.exp(-(x**2 + nu**2)/(2*sigma2)) *
             mp.besseli(0, x*nu/sigma2))
        return p


def logpdf(x, nu, sigma):
    """
    Logarithm of the PDF for the Rice distribution.
    """
    with mp.extradps(5):
        nu, sigma = _validate_params(nu, sigma)
        x = mp.mpf(x)
        if x <= 0:
            return mp.ninf
        sigma2 = sigma**2
        # p = ((x / sigma2) * mp.exp(-(x**2 + nu**2)/(2*sigma2)) *
        #      mp.besseli(0, x*nu/sigma2))
        # return p
        logp = (mp.log(x) - 2*mp.log(sigma) - (x**2 + nu**2)/(2*sigma2)
                + mp.log(mp.besseli(0, x*nu/sigma2)))
        return logp


def cdf(x, nu, sigma):
    """
    CDF for the Rice distribution.
    """
    with mp.extradps(5):
        nu, sigma = _validate_params(nu, sigma)
        x = mp.mpf(x)
        if x <= 0:
            return mp.zero
        c = cmarcumq(1, nu/sigma, x/sigma)
        return c


def sf(x, nu, sigma):
    """
    Survival function for the Rice distribution.
    """
    with mp.extradps(5):
        nu, sigma = _validate_params(nu, sigma)
        x = mp.mpf(x)
        if x <= 0:
            return mp.one
        s = marcumq(1, nu/sigma, x/sigma)
        return s


def mean(nu, sigma):
    """
    Mean of the Rice distribution.
    """
    with mp.extradps(5):
        nu, sigma = _validate_params(nu, sigma)
        return (sigma * mp.sqrt(mp.pi/2)
                * mp.laguerre(0.5, 0, -(nu/sigma)**2/2))
