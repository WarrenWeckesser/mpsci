"""
Half-logistic distribution
--------------------------

See https://en.wikipedia.org/wiki/Half-logistic_distribution.

This implementation includes location and scale parameters.

"""

from mpmath import mp
from ._common import _validate_p, _validate_loc_scale


__all__ = ['pdf', 'logpdf', 'cdf', 'invcdf', 'sf', 'invsf',
           'support']


@mp.extradps(5)
def pdf(x, loc=0, scale=1):
    """
    Probability density function for the half-logistic distribution.
    """
    loc, scale = _validate_loc_scale(loc, scale)
    x = mp.mpf(x)
    if x < loc:
        return mp.zero
    z = (x - loc)/scale
    return mp.sech(z/2)**2 / 2 / scale


@mp.extradps(5)
def logpdf(x, loc=0, scale=1):
    """
    Logarithm of the PDF for the half-logistic distribution.
    """
    loc, scale = _validate_loc_scale(loc, scale)
    x = mp.mpf(x)
    if x < loc:
        return mp.ninf
    z = (x - loc)/scale
    return 2*mp.log(mp.sech(z/2)) - mp.log(2) - mp.log(scale)


@mp.extradps(5)
def cdf(x, loc=0, scale=1):
    """
    Cumulative distribution function for the half-logistic distribution.
    """
    loc, scale = _validate_loc_scale(loc, scale)
    x = mp.mpf(x)
    if x < loc:
        return mp.zero
    z = (x - loc)/scale
    return -mp.expm1(-z)/(1 + mp.exp(-z))


@mp.extradps(5)
def invcdf(p, loc=0, scale=1):
    """
    Inverse of the CDF for the half-logistic distribution.

    This function is also known as the *quantile function*.
    """
    loc, scale = _validate_loc_scale(loc, scale)
    p = _validate_p(p)
    if p == 1:
        return mp.inf
    if p == 0:
        return loc
    return loc - scale*(mp.log1p(-p) - mp.log1p(p))


@mp.extradps(5)
def sf(x, loc=0, scale=1):
    """
    Survival function for the half-logistic distribution.
    """
    loc, scale = _validate_loc_scale(loc, scale)
    x = mp.mpf(x)
    if x < loc:
        return mp.one
    z = (x - loc)/scale
    return 2*mp.exp(-z)/(1 + mp.exp(-z))


@mp.extradps(5)
def invsf(p, loc=0, scale=1):
    """
    Inverse of the survival function for the half-logistic distribution.
    """
    loc, scale = _validate_loc_scale(loc, scale)
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
    loc, scale = _validate_loc_scale(loc, scale)
    return (loc, mp.inf)
