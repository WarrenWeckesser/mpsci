"""
Slash distribution
------------------

See https://en.wikipedia.org/wiki/Slash_distribution for details.

"""

import mpmath
from ._common import _validate_p


__all__ = ['pdf', 'logpdf', 'cdf', 'sf', 'invcdf', 'invsf']


def pdf(x):
    """
    Probability density function of the slash distribution.
    """
    with mpmath.extradps(5):
        if x == 0:
            return 1/(2*mpmath.sqrt(2*mpmath.pi))
        x = mpmath.mpf(x)
        return (mpmath.npdf(0) - mpmath.npdf(x))/x**2


def logpdf(x):
    """
    Natural logarithm of the PDF of the slash distribution.
    """
    with mpmath.extradps(5):
        if x == 0:
            return mpmath.log(pdf(0))
        x = mpmath.mpf(x)
        return (mpmath.log(mpmath.npdf(0) - mpmath.npdf(x))
                - 2*mpmath.log(mpmath.absmax(x)))


def cdf(x):
    """
    Cumulative distribution function for the slash distribution.
    """
    with mpmath.extradps(5):
        if x == 0:
            return mpmath.mp.one/2
        x = mpmath.mpf(x)
        return mpmath.ncdf(x) - (mpmath.npdf(0) - mpmath.npdf(x))/x


def sf(x):
    """
    Survival function for the slash distribution.
    """
    with mpmath.extradps(5):
        if x == 0:
            return mpmath.mp.one/2
        x = mpmath.mpf(x)
        return mpmath.ncdf(-x) + (mpmath.npdf(0) - mpmath.npdf(x))/x


_npdf0 = mpmath.npdf(0)


def invcdf(p):
    """
    Inverse of the CDF of the slash distribution.

    Also known as the quantile function.

    This function numerically inverts the CDF function so it
    may be slow, and in some cases it may fail to find a solution.
    """
    with mpmath.extradps(5):
        p = _validate_p(p)
        if p == 0:
            return mpmath.ninf
        if p == 1:
            return mpmath.inf
        if p == 0.5:
            return mpmath.mp.zero
        if p > 0.5:
            x0 = _npdf0/(1 - p)
        else:
            x0 = -_npdf0/p
        return mpmath.findroot(lambda x: cdf(x) - p, x0=x0)


def invsf(p):
    """
    Inverse of the survival function of the slash distribution.

    This function numerically inverts the survival function so it
    may be slow, and in some cases it may fail to find a solution.
    """
    with mpmath.extradps(5):
        p = _validate_p(p)
        if p == 0:
            return mpmath.inf
        if p == 1:
            return mpmath.ninf
        if p == 0.5:
            return mpmath.mp.zero
        if p > 0.5:
            x0 = -_npdf0/(1 - p)
        else:
            x0 = _npdf0/p
        return mpmath.findroot(lambda x: sf(x) - p, x0=x0)
