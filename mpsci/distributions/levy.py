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

from mpmath import mp
from ._common import _validate_p, _validate_loc_scale


__all__ = ['logpdf', 'pdf', 'cdf', 'invcdf', 'sf', 'invsf',
           'support']


def _erfcinv(y):
    with mp.extradps(5):
        return mp.erfinv(1 - y)


def logpdf(x, mu=0, sigma=1):
    """
    Log of the PDF of the Lévy distribution.
    """
    with mp.extradps(5):
        x = mp.mpf(x)
        mu, sigma = _validate_loc_scale(mu, sigma, scale_name='sigma')
        if x <= mu:
            return mp.ninf
        t1 = mp.log(sigma)/2 - mp.log(2*mp.pi)/2
        t2 = -sigma / (2*(x - mu))
        t3 = mp.log(x - mu)*3/2
        return t1 + t2 - t3


def pdf(x, mu=0, sigma=1):
    """
    PDF of the Lévy distribution.
    """
    return mp.exp(logpdf(x, mu, sigma))


def cdf(x, mu=0, sigma=1):
    """
    CDF of the Lévy distribution.
    """
    with mp.extradps(5):
        x = mp.mpf(x)
        mu, sigma = _validate_loc_scale(mu, sigma, scale_name='sigma')
        if x <= mu:
            return mp.zero
        arg = mp.sqrt(sigma / (2*(x - mu)))
        return mp.erfc(arg)


def invcdf(p, mu=0, sigma=1):
    """
    Inverse of the CDF of the Lévy distribution.
    """
    with mp.extradps(5):
        p = _validate_p(p)
        mu, sigma = _validate_loc_scale(mu, sigma, scale_name='sigma')
        return mu + sigma / (2*_erfcinv(p)**2)


def sf(x, mu=0, sigma=1):
    """
    Survival function of the Lévy distribution.
    """
    with mp.extradps(5):
        x = mp.mpf(x)
        mu, sigma = _validate_loc_scale(mu, sigma, scale_name='sigma')
        if x <= mu:
            return mp.one
        arg = mp.sqrt(sigma / (2*(x - mu)))
        return mp.erf(arg)


def invsf(p, mu=0, sigma=1):
    """
    Inverse of the survivial function of the Lévy distribution.
    """

    with mp.extradps(5):
        p = _validate_p(p)
        mu, sigma = _validate_loc_scale(mu, sigma, scale_name='sigma')
        return mu + sigma / (2*mp.erfinv(p)**2)


def support(p, mu=0, sigma=1):
    """
    Support of the Lévy distribution.
    """
    with mp.extradps(5):
        p = _validate_p(p)
        mu, sigma = _validate_loc_scale(mu, sigma, scale_name='sigma')
        return (mu, mp.inf)
