"""
Folded normal distribution
--------------------------

The parameters mu and sigma are the mean and standard deviation
of the underlying normal distribution *before folding*. The support is
x >= 0.

See https://en.wikipedia.org/wiki/Folded_normal_distribution

This parametrization is not the same as the parametrization used
in SciPy. The conversion from ``folded_normal`` parameters mu and
sigma to SciPy's ``c``, ``loc`` and ``scale``::

    c = mu/sigma
    loc = 0
    scale = sigma
"""

from mpmath import mp
from .normal import logpdf as normal_logpdf
from ..fun import logsumexp


__all__ = ['pdf', 'logpdf', 'cdf', 'sf', 'mean', 'var']


def _validate_sigma(sigma):
    if sigma <= 0:
        raise ValueError('sigma must be positive')
    return mp.mpf(sigma)


def pdf(x, mu, sigma):
    """
    Probability density function of the folded normal distribution.
    """
    with mp.extradps(5):
        x = mp.mpf(x)
        mu = mp.mpf(mu)
        sigma = _validate_sigma(sigma)
        if x < 0:
            return mp.zero
        return mp.npdf(x, mu, sigma) + mp.npdf(-x, mu, sigma)


def logpdf(x, mu, sigma):
    """
    Logarithm of the PDF of the folded normal distribution.
    """
    with mp.extradps(5):
        x = mp.mpf(x)
        mu = mp.mpf(mu)
        sigma = _validate_sigma(sigma)
        if x < 0:
            return mp.ninf
        logp1 = normal_logpdf(x, mu, sigma)
        logp2 = normal_logpdf(-x, mu, sigma)
        return logsumexp([logp1, logp2])


def cdf(x, mu, sigma):
    """
    Cumulative distribution function of the folded normal distribution.
    """
    with mp.extradps(5):
        x = mp.mpf(x)
        mu = mp.mpf(mu)
        sigma = _validate_sigma(sigma)
        if x < 0:
            return mp.zero
        return mp.ncdf(x, mu, sigma) - mp.ncdf(-x, mu, sigma)


def sf(x, mu, sigma):
    """
    Survival function of the folded normal distribution.
    """
    with mp.extradps(5):
        x = mp.mpf(x)
        mu = mp.mpf(mu)
        sigma = _validate_sigma(sigma)
        if x < 0:
            return mp.one
        return mp.ncdf(-x, -mu, sigma) + mp.ncdf(-x, mu, sigma)


def mean(mu, sigma):
    """
    Mean of the folded normal distribution.
    """
    with mp.extradps(5):
        mu = mp.mpf(mu)
        sigma = _validate_sigma(sigma)
        r = mu/sigma
        m = sigma*mp.sqrt(2/mp.pi)*mp.exp(-r**2/2) + mu*(1 - 2*mp.ncdf(-r))
        return m


def var(mu, sigma):
    """
    Variance of the folded normal distribution.
    """
    with mp.extradps(5):
        mu = mp.mpf(mu)
        sigma = _validate_sigma(sigma)
        return mu**2 + sigma**2 - mean(mu, sigma)**2
