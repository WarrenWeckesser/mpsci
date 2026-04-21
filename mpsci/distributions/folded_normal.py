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


__all__ = ['pdf', 'logpdf', 'cdf', 'sf', 'mean', 'mode', 'var',
           'support']


def _validate_sigma(sigma):
    if sigma <= 0:
        raise ValueError('sigma must be positive')
    return mp.mpf(sigma)


@mp.extradps(5)
def pdf(x, mu, sigma):
    """
    Probability density function of the folded normal distribution.
    """
    x = mp.mpf(x)
    mu = mp.mpf(mu)
    sigma = _validate_sigma(sigma)
    if x < 0:
        return mp.zero
    return mp.npdf(x, mu, sigma) + mp.npdf(-x, mu, sigma)


@mp.extradps(5)
def logpdf(x, mu, sigma):
    """
    Logarithm of the PDF of the folded normal distribution.
    """
    x = mp.mpf(x)
    mu = mp.mpf(mu)
    sigma = _validate_sigma(sigma)
    if x < 0:
        return mp.ninf
    logp1 = normal_logpdf(x, mu, sigma)
    logp2 = normal_logpdf(-x, mu, sigma)
    return logsumexp([logp1, logp2])


@mp.extradps(5)
def cdf(x, mu, sigma):
    """
    Cumulative distribution function of the folded normal distribution.
    """
    x = mp.mpf(x)
    mu = mp.mpf(mu)
    sigma = _validate_sigma(sigma)
    if x < 0:
        return mp.zero
    return mp.ncdf(x, mu, sigma) - mp.ncdf(-x, mu, sigma)


@mp.extradps(5)
def sf(x, mu, sigma):
    """
    Survival function of the folded normal distribution.
    """
    x = mp.mpf(x)
    mu = mp.mpf(mu)
    sigma = _validate_sigma(sigma)
    if x < 0:
        return mp.one
    return mp.ncdf(-x, -mu, sigma) + mp.ncdf(-x, mu, sigma)


def support(mu, sigma):
    """
    Support of the folded normal distribution.
    """
    mu = mp.mpf(mu)
    sigma = _validate_sigma(sigma)
    return (mp.zero, mp.inf)


@mp.extradps(5)
def mean(mu, sigma):
    """
    Mean of the folded normal distribution.
    """
    mu = mp.mpf(mu)
    sigma = _validate_sigma(sigma)
    r = mu/sigma
    m = sigma*mp.sqrt(2/mp.pi)*mp.exp(-r**2/2) + mu*(1 - 2*mp.ncdf(-r))
    return m


def _mode_eq(t, r):
    return t - r * mp.tanh(r*t)


def _mode_getstart(r):
    x = r
    while _mode_eq(x, r) >= 0:
        x = x/2
    return x


@mp.extradps(5)
def mode(mu, sigma):
    """
    Mode of the folded normal distribution.

    If abs(mu) <= sigma, then the mode is 0.
    """
    mu = abs(mp.mpf(mu))
    sigma = _validate_sigma(sigma)
    if mu <= sigma:
        return mp.zero
    r = mu/sigma
    # TODO: It should be possible to derive a better calculation of t0.
    if r < 1.25:
        t0 = _mode_getstart(r)
    else:
        t0 = r
    t = mp.findroot(lambda s: s - r*mp.tanh(r*s), t0)
    return sigma * t


@mp.extradps(5)
def var(mu, sigma):
    """
    Variance of the folded normal distribution.
    """
    mu = mp.mpf(mu)
    sigma = _validate_sigma(sigma)
    return mu**2 + sigma**2 - mean(mu, sigma)**2
