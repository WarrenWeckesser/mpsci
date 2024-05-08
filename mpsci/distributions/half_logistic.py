"""
Half-logistic distribution
--------------------------

See https://en.wikipedia.org/wiki/Half-logistic_distribution.

This implementation includes location and scale parameters.

"""

from mpmath import mp
from ._common import _validate_p


__all__ = ['pdf', 'logpdf', 'cdf', 'invcdf', 'sf', 'invsf',
           'support']


def _validate_params(loc, scale):
    if scale <= 0:
        raise ValueError('scale must be greater than 0')
    return mp.mpf(loc), mp.mpf(scale)


def pdf(x, loc=0, scale=1):
    """
    Probability density function for the half-logistic distribution.
    """
    with mp.extradps(5):
        loc, scale = _validate_params(loc, scale)
        x = mp.mpf(x)
        if x < loc:
            return mp.zero
        z = (x - loc)/scale
        return mp.sech(z/2)**2 / 2 / scale


def logpdf(x, loc=0, scale=1):
    """
    Logarithm of the PDF for the half-logistic distribution.
    """
    with mp.extradps(5):
        loc, scale = _validate_params(loc, scale)
        x = mp.mpf(x)
        if x < loc:
            return mp.ninf
        z = (x - loc)/scale
        return 2*mp.log(mp.sech(z/2)) - mp.log(2) - mp.log(scale)


def cdf(x, loc=0, scale=1):
    """
    Cumulative distribution function for the half-logistic distribution.
    """
    with mp.extradps(5):
        loc, scale = _validate_params(loc, scale)
        x = mp.mpf(x)
        if x < loc:
            return mp.zero
        z = (x - loc)/scale
        return -mp.expm1(-z)/(1 + mp.exp(-z))


def invcdf(p, loc=0, scale=1):
    """
    Inverse of the CDF for the half-logistic distribution.

    This function is also known as the *quantile function*.
    """
    with mp.extradps(5):
        loc, scale = _validate_params(loc, scale)
        p = _validate_p(p)
        if p == 1:
            return mp.inf
        if p == 0:
            return loc
        return loc - scale*(mp.log1p(-p) - mp.log1p(p))


def sf(x, loc=0, scale=1):
    """
    Survival function for the half-logistic distribution.
    """
    with mp.extradps(5):
        loc, scale = _validate_params(loc, scale)
        x = mp.mpf(x)
        if x < loc:
            return mp.one
        z = (x - loc)/scale
        return 2*mp.exp(-z)/(1 + mp.exp(-z))


def invsf(p, loc=0, scale=1):
    """
    Inverse of the survival function for the half-logistic distribution.
    """
    with mp.extradps(5):
        loc, scale = _validate_params(loc, scale)
        p = _validate_p(p)
        if p == 1:
            return loc
        if p == 0:
            return mp.inf
        return loc - scale*mp.log(p/2/(1 - p/2))


def support(loc=0, scale=1):
    """
    Support of the half-logistic distribution.
    """
    with mp.extradps(5):
        loc, scale = _validate_params(loc, scale)
        return (loc, mp.inf)
