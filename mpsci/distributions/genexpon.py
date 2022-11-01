"""
Generalized exponential distribution
------------------------------------

This is the same distribution as `scipy.stats.genexpon`.
"""

import mpmath


__all__ = ['pdf', 'logpdf', 'cdf', 'sf', 'invcdf', 'invsf']


def _validate_params(a, b, c, loc, scale):
    if a <= 0:
        raise ValueError("'a' must be greater than 0.")
    if b <= 0:
        raise ValueError("'b' must be greater than 0.")
    if c <= 0:
        raise ValueError("'c' must be greater than 0.")
    if scale <= 0:
        raise ValueError("'scale' must be greater than 0.")
    return (mpmath.mpf(t) for t in [a, b, c, loc, scale])


def _validate_x_params(x, a, b, c, loc, scale):
    x = mpmath.mpf(x)
    a, b, c, loc, scale = _validate_params(a, b, c, loc, scale)
    if x < loc:
        raise ValueError("'x' must not be less than 'loc'.")
    return x, a, b, c, loc, scale


def pdf(x, a, b, c, loc=0, scale=1):
    """
    PDF of the generalized exponential distribution.
    """
    with mpmath.extradps(5):
        x, a, b, c, loc, scale = _validate_x_params(x, a, b, c, loc, scale)
        z = (x - loc) / scale
        s = a + b
        r = b / c
        p = ((a + b*(-mpmath.expm1(-c*z))) *
             mpmath.exp(-s*z + r*(-mpmath.expm1(-c*z))))/scale
    return p


def logpdf(x, a, b, c, loc=0, scale=1):
    """
    Natural logarithm of the PDF of the generalized exponential distribution.
    """
    with mpmath.extradps(5):
        x, a, b, c, loc, scale = _validate_x_params(x, a, b, c, loc, scale)
        z = (x - loc) / scale
        s = a + b
        r = b / c
        logp = (mpmath.log(a + b*(-mpmath.expm1(-c*z))) +
                (-s*z + r*(-mpmath.expm1(-c*z))))
    return logp


def cdf(x, a, b, c, loc=0, scale=1):
    """
    CDF of the generalized exponential distribution.
    """
    with mpmath.extradps(5):
        x, a, b, c, loc, scale = _validate_x_params(x, a, b, c, loc, scale)
        z = (x - loc) / scale
        s = a + b
        r = b / c
        result = -mpmath.expm1(-s*z + r*(-mpmath.expm1(-c*z)))
    return result


def sf(x, a, b, c, loc=0, scale=1):
    """
    Survival function of the generalized exponential distribution.
    """
    with mpmath.extradps(5):
        x, a, b, c, loc, scale = _validate_x_params(x, a, b, c, loc, scale)
        z = (x - loc) / scale
        s = a + b
        r = b / c
        result = mpmath.exp(-s*z + r*(-mpmath.expm1(-c*z)))
    return result


def invcdf(p, a, b, c, loc=0, scale=1):
    """
    Inverse of the CDF of the generalized exponential distribution.

    This is also known as the quantile function.
    """
    if p < 0 or p > 1:
        raise ValueError("'p' must be between 0 and 1.")
    with mpmath.extradps(5):
        p = mpmath.mpf(p)
        a, b, c, loc, scale = _validate_params(a, b, c, loc, scale)
        r = b/(a + b)
        s = r/c - mpmath.log1p(-p)/(a + b)
        x = s + mpmath.lambertw(-r*mpmath.exp(-s*c))/c
    return loc + scale*x


def invsf(p, a, b, c, loc=0, scale=1):
    """
    Inverse of the survival function of the gen. exponential distribution.
    """
    if p < 0 or p > 1:
        raise ValueError("'p' must be between 0 and 1.")
    with mpmath.extradps(5):
        p = mpmath.mpf(p)
        a, b, c, loc, scale = _validate_params(a, b, c, loc, scale)
        r = b/(a + b)
        s = r/c - mpmath.log(p)/(a + b)
        x = s + mpmath.lambertw(-r*mpmath.exp(-s*c))/c
    return loc + scale*x
