"""
Generalized exponential distribution
------------------------------------

This is the same distribution as `scipy.stats.genexpon`.
"""

from mpmath import mp
from ._common import _validate_p


__all__ = ['pdf', 'logpdf', 'cdf', 'sf', 'invcdf', 'invsf',
           'support']


def _validate_params(a, b, c, loc, scale):
    if a <= 0:
        raise ValueError("'a' must be greater than 0.")
    if b <= 0:
        raise ValueError("'b' must be greater than 0.")
    if c <= 0:
        raise ValueError("'c' must be greater than 0.")
    if scale <= 0:
        raise ValueError("'scale' must be greater than 0.")
    return (mp.mpf(t) for t in [a, b, c, loc, scale])


def _validate_x_params(x, a, b, c, loc, scale):
    x = mp.mpf(x)
    a, b, c, loc, scale = _validate_params(a, b, c, loc, scale)
    if x < loc:
        raise ValueError("'x' must not be less than 'loc'.")
    return x, a, b, c, loc, scale


@mp.extradps(5)
def pdf(x, a, b, c, loc=0, scale=1):
    """
    PDF of the generalized exponential distribution.
    """
    x, a, b, c, loc, scale = _validate_x_params(x, a, b, c, loc, scale)
    z = (x - loc) / scale
    s = a + b
    r = b / c
    return ((a + b*(-mp.expm1(-c*z))) *
            mp.exp(-s*z + r*(-mp.expm1(-c*z)))) / scale


@mp.extradps(5)
def logpdf(x, a, b, c, loc=0, scale=1):
    """
    Natural logarithm of the PDF of the generalized exponential distribution.
    """
    x, a, b, c, loc, scale = _validate_x_params(x, a, b, c, loc, scale)
    z = (x - loc) / scale
    s = a + b
    r = b / c
    return mp.log(a + b*(-mp.expm1(-c*z))) + (-s*z + r*(-mp.expm1(-c*z)))


@mp.extradps(5)
def cdf(x, a, b, c, loc=0, scale=1):
    """
    CDF of the generalized exponential distribution.
    """
    x, a, b, c, loc, scale = _validate_x_params(x, a, b, c, loc, scale)
    z = (x - loc) / scale
    s = a + b
    r = b / c
    return -mp.expm1(-s*z + r*(-mp.expm1(-c*z)))


@mp.extradps(5)
def sf(x, a, b, c, loc=0, scale=1):
    """
    Survival function of the generalized exponential distribution.
    """
    x, a, b, c, loc, scale = _validate_x_params(x, a, b, c, loc, scale)
    z = (x - loc) / scale
    s = a + b
    r = b / c
    return mp.exp(-s*z + r*(-mp.expm1(-c*z)))


@mp.extradps(5)
def invcdf(p, a, b, c, loc=0, scale=1):
    """
    Inverse of the CDF of the generalized exponential distribution.

    This is also known as the quantile function.
    """
    p = _validate_p(p)
    a, b, c, loc, scale = _validate_params(a, b, c, loc, scale)
    r = b/(a + b)
    s = r/c - mp.log1p(-p)/(a + b)
    x = s + mp.lambertw(-r*mp.exp(-s*c))/c
    return loc + scale*x


@mp.extradps(5)
def invsf(p, a, b, c, loc=0, scale=1):
    """
    Inverse of the survival function of the gen. exponential distribution.
    """
    p = _validate_p(p)
    a, b, c, loc, scale = _validate_params(a, b, c, loc, scale)
    r = b/(a + b)
    s = r/c - mp.log(p)/(a + b)
    x = s + mp.lambertw(-r*mp.exp(-s*c))/c
    return loc + scale*x


def support(a, b, c, loc=0, scale=1):
    """
    Support of the generalized exponential distribution.
    """
    a, b, c, loc, scale = _validate_params(a, b, c, loc, scale)
    return (loc, mp.inf)
