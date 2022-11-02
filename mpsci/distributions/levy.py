"""
Lévy Distribution
-----------------

The parameters of the distribution are mu (the location) and
sigma (the scale).

See the wikipedia article "Lévy distribution" [1]_ for more
information.

.. [1] Lévy distribution,
       https://en.wikipedia.org/wiki/L%C3%A9vy_distribution

"""

import mpmath
from ._common import _validate_p


__all__ = ['logpdf', 'pdf', 'cdf', 'invcdf', 'sf', 'invsf']


def _erfcinv(y):
    with mpmath.extradps(5):
        return mpmath.erfinv(1 - y)


def logpdf(x, mu=0, sigma=1):
    """
    Log of the PDF of the Lévy distribution.
    """
    if sigma <= 0:
        raise ValueError('sigma must be positive.')

    if x <= mu:
        return mpmath.ninf

    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        mu = mpmath.mpf(mu)
        sigma = mpmath.mpf(sigma)
        t1 = mpmath.log(sigma)/2 - mpmath.log(2*mpmath.pi)/2
        t2 = -sigma / (2*(x - mu))
        t3 = mpmath.log(x - mu)*3/2
        return t1 + t2 - t3


def pdf(x, mu=0, sigma=1):
    """
    PDF of the Lévy distribution.
    """
    return mpmath.exp(logpdf(x, mu, sigma))


def cdf(x, mu=0, sigma=1):
    """
    CDF of the Lévy distribution.
    """
    if sigma <= 0:
        raise ValueError('sigma must be positive.')

    if x <= mu:
        return mpmath.zero

    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        mu = mpmath.mpf(mu)
        sigma = mpmath.mpf(sigma)
        arg = mpmath.sqrt(sigma / (2*(x - mu)))
        return mpmath.erfc(arg)


def invcdf(p, mu=0, sigma=1):
    """
    Inverse of the CDF of the Lévy distribution.
    """
    if sigma <= 0:
        raise ValueError('sigma must be positive.')
    with mpmath.extradps(5):
        p = _validate_p(p)
        mu = mpmath.mpf(mu)
        sigma = mpmath.mpf(sigma)
        return mu + sigma / (2*_erfcinv(p)**2)


def sf(x, mu=0, sigma=1):
    """
    Survival function of the Lévy distribution.
    """
    if sigma <= 0:
        raise ValueError('sigma must be positive.')

    if x <= mu:
        return mpmath.one

    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        mu = mpmath.mpf(mu)
        sigma = mpmath.mpf(sigma)
        arg = mpmath.sqrt(sigma / (2*(x - mu)))
        return mpmath.erf(arg)


def invsf(p, mu=0, sigma=1):
    """
    Inverse of the survivial function of the Lévy distribution.
    """
    if sigma <= 0:
        raise ValueError('sigma must be positive.')
    with mpmath.extradps(5):
        p = _validate_p(p)
        mu = mpmath.mpf(mu)
        sigma = mpmath.mpf(sigma)
        return mu + sigma / (2*mpmath.erfinv(p)**2)
