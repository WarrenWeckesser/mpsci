"""
Logistic distribution
---------------------

The logistic distribution is also known as the sech-squared distribution.

"""

import mpmath


__all__ = ['pdf', 'cdf', 'sf', 'invcdf', 'invsf', 'mean', 'var']


def pdf(x, loc=0, scale=1):
    """
    PDF of the logistic distribution.
    """
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        loc = mpmath.mpf(loc)
        scale = mpmath.mpf(scale)
        z = (x - loc) / scale
        p = mpmath.sech(z/2)**2 / (4*scale)
    return p


def cdf(x, loc=0, scale=1):
    """
    CDF of the logistic distribution.
    """
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        loc = mpmath.mpf(loc)
        scale = mpmath.mpf(scale)
        z = (x - loc) / scale
        p = (1 + mpmath.tanh(z/2)) / 2
    return p


def sf(x, loc=0, scale=1):
    """
    Survival function of the logistic distribution.
    """
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        loc = mpmath.mpf(loc)
        scale = mpmath.mpf(scale)
        z = (x - loc) / scale
        p = (1 - mpmath.tanh(z/2)) / 2
    return p


def invcdf(p, loc=0, scale=1):
    """
    Inverse CDF of the logistic distribution.

    This function is also known as the quantile function or the percent
    point function.
    """
    with mpmath.extradps(5):
        p = mpmath.mpf(p)
        loc = mpmath.mpf(loc)
        scale = mpmath.mpf(scale)
        x = loc + scale*(mpmath.log(p) - mpmath.log1p(-p))
    return x


def invsf(p, loc=0, scale=1):
    """
    Inverse survival function of the logistic distribution.
    """
    with mpmath.extradps(5):
        p = mpmath.mpf(p)
        loc = mpmath.mpf(loc)
        scale = mpmath.mpf(scale)
        x = loc + scale*(mpmath.log1p(-p) - mpmath.log(p))
    return x


def mean(loc=0, scale=1):
    """
    Mean of the logistic distribution.
    """
    with mpmath.extradps(5):
        return mpmath.mpf(loc)


def var(loc=0, scale=1):
    """
    Variance of the logistic distribution.
    """
    with mpmath.extradps(5):
        scale = mpmath.mpf(scale)
        return scale**2 * mpmath.pi**2 / 3