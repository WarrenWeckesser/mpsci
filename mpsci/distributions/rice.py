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
from ._common import _validate_moment_n


__all__ = ['pdf', 'logpdf', 'cdf', 'sf',
           'support', 'mean', 'var', 'noncentral_moment']


def _validate_params(nu, sigma):
    if nu < 0:
        raise ValueError('nu must be nonnegative')
    if sigma <= 0:
        raise ValueError('sigma must be greater than 0')
    return mp.mpf(nu), mp.mpf(sigma)


@mp.extradps(5)
def pdf(x, nu, sigma):
    """
    PDF for the Rice distribution.
    """
    nu, sigma = _validate_params(nu, sigma)
    x = mp.mpf(x)
    if x <= 0:
        return mp.zero
    sigma2 = sigma**2
    p = ((x / sigma2) * mp.exp(-(x**2 + nu**2)/(2*sigma2)) *
         mp.besseli(0, x*nu/sigma2))
    return p


@mp.extradps(5)
def logpdf(x, nu, sigma):
    """
    Logarithm of the PDF for the Rice distribution.
    """
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


@mp.extradps(5)
def cdf(x, nu, sigma):
    """
    CDF for the Rice distribution.
    """
    nu, sigma = _validate_params(nu, sigma)
    x = mp.mpf(x)
    if x <= 0:
        return mp.zero
    c = cmarcumq(1, nu/sigma, x/sigma)
    return c


@mp.extradps(5)
def sf(x, nu, sigma):
    """
    Survival function for the Rice distribution.
    """
    nu, sigma = _validate_params(nu, sigma)
    x = mp.mpf(x)
    if x <= 0:
        return mp.one
    s = marcumq(1, nu/sigma, x/sigma)
    return s


@mp.extradps(5)
def support(nu, sigma):
    """
    Support of the Rice distribution.
    """
    nu, sigma = _validate_params(nu, sigma)
    return (mp.zero, mp.inf)


@mp.extradps(5)
def mean(nu, sigma):
    """
    Mean of the Rice distribution.
    """
    nu, sigma = _validate_params(nu, sigma)
    return (sigma * mp.sqrt(mp.pi/2)
            * mp.laguerre(0.5, 0, -(nu/sigma)**2/2))


@mp.extradps(5)
def var(nu, sigma):
    """
    Variance of the Rice distribution.
    """
    nu, sigma = _validate_params(nu, sigma)
    return noncentral_moment(2, nu, sigma) - mean(nu, sigma)**2


@mp.extradps(5)
def noncentral_moment(n, nu, sigma):
    """
    Noncentral moment of the Rice distribution.

    The value is also known as the raw moment.
    """
    n = _validate_moment_n(n)
    nu, sigma = _validate_params(nu, sigma)
    if n == 0:
        return mp.one
    t1 = n*mp.log(sigma)
    t2 = (n/2)*mp.log(2)
    t3 = mp.loggamma(1 + n/2)
    t4 = mp.log(mp.hyp1f1(-n/2, 1, -(nu/sigma)**2/2))
    return mp.exp(mp.fsum([t1, t2, t3, t4]))
