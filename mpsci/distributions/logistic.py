"""
Logistic distribution
---------------------

The logistic distribution is also known as the sech-squared distribution.

"""

from mpmath import mp
from mpsci.stats import mean as _mean


__all__ = ['pdf', 'logpdf', 'cdf', 'sf', 'invcdf', 'invsf', 'mean', 'var',
           'mom']


def pdf(x, loc=0, scale=1):
    """
    PDF of the logistic distribution.
    """
    with mp.extradps(5):
        x = mp.mpf(x)
        loc = mp.mpf(loc)
        scale = mp.mpf(scale)
        z = (x - loc) / scale
        p = mp.sech(z/2)**2 / (4*scale)
    return p


def logpdf(x, loc=0, scale=1):
    """
    Logarithm of the PDF of the logistic distribution.
    """
    with mp.extradps(5):
        x = mp.mpf(x)
        loc = mp.mpf(loc)
        scale = mp.mpf(scale)
        z = (x - loc) / scale
        logp = 2*mp.log(mp.sech(z/2)) - mp.log(4*scale)
    return logp


def cdf(x, loc=0, scale=1):
    """
    CDF of the logistic distribution.
    """
    with mp.extradps(5):
        x = mp.mpf(x)
        loc = mp.mpf(loc)
        scale = mp.mpf(scale)
        z = (x - loc) / scale
        p = (1 + mp.tanh(z/2)) / 2
    return p


def sf(x, loc=0, scale=1):
    """
    Survival function of the logistic distribution.
    """
    with mp.extradps(5):
        x = mp.mpf(x)
        loc = mp.mpf(loc)
        scale = mp.mpf(scale)
        z = (x - loc) / scale
        p = (1 - mp.tanh(z/2)) / 2
    return p


def invcdf(p, loc=0, scale=1):
    """
    Inverse CDF of the logistic distribution.

    This function is also known as the quantile function or the percent
    point function.
    """
    with mp.extradps(5):
        p = mp.mpf(p)
        loc = mp.mpf(loc)
        scale = mp.mpf(scale)
        x = loc + scale*(mp.log(p) - mp.log1p(-p))
    return x


def invsf(p, loc=0, scale=1):
    """
    Inverse survival function of the logistic distribution.
    """
    with mp.extradps(5):
        p = mp.mpf(p)
        loc = mp.mpf(loc)
        scale = mp.mpf(scale)
        x = loc + scale*(mp.log1p(-p) - mp.log(p))
    return x


def mean(loc=0, scale=1):
    """
    Mean of the logistic distribution.
    """
    with mp.extradps(5):
        return mp.mpf(loc)


def var(loc=0, scale=1):
    """
    Variance of the logistic distribution.
    """
    with mp.extradps(5):
        scale = mp.mpf(scale)
        return scale**2 * mp.pi**2 / 3


def mom(x):
    """
    Method of moments parameter estimation for the logistic distribution.

    x must be a sequence of numbers.

    Returns (loc, scale).
    """
    with mp.extradps(5):
        M1 = _mean(x)
        M2 = _mean([mp.mpf(t)**2 for t in x])
        return M1, mp.sqrt(3*(M2 - M1**2))/mp.pi
