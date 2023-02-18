"""
Slash distribution
------------------

See https://en.wikipedia.org/wiki/Slash_distribution for details.

"""

from mpmath import mp
from ._common import _validate_p


__all__ = ['pdf', 'logpdf', 'cdf', 'sf', 'invcdf', 'invsf']


def pdf(x):
    """
    Probability density function of the slash distribution.
    """
    with mp.extradps(5):
        if x == 0:
            return 1/(2*mp.sqrt(2*mp.pi))
        x = mp.mpf(x)
        return (mp.npdf(0) - mp.npdf(x))/x**2


def logpdf(x):
    """
    Natural logarithm of the PDF of the slash distribution.
    """
    with mp.extradps(5):
        if x == 0:
            return mp.log(pdf(0))
        x = mp.mpf(x)
        return (mp.log(mp.npdf(0) - mp.npdf(x))
                - 2*mp.log(mp.absmax(x)))


def cdf(x):
    """
    Cumulative distribution function for the slash distribution.
    """
    with mp.extradps(5):
        if x == 0:
            return mp.one/2
        x = mp.mpf(x)
        return mp.ncdf(x) - (mp.npdf(0) - mp.npdf(x))/x


def sf(x):
    """
    Survival function for the slash distribution.
    """
    with mp.extradps(5):
        if x == 0:
            return mp.one/2
        x = mp.mpf(x)
        return mp.ncdf(-x) + (mp.npdf(0) - mp.npdf(x))/x


_npdf0 = mp.npdf(0)


def invcdf(p):
    """
    Inverse of the CDF of the slash distribution.

    Also known as the quantile function.

    This function numerically inverts the CDF function so it
    may be slow, and in some cases it may fail to find a solution.
    """
    with mp.extradps(5):
        p = _validate_p(p)
        if p == 0:
            return mp.ninf
        if p == 1:
            return mp.inf
        if p == 0.5:
            return mp.zero
        if p > 0.5:
            x0 = _npdf0/(1 - p)
        else:
            x0 = -_npdf0/p
        return mp.findroot(lambda x: cdf(x) - p, x0=x0)


def invsf(p):
    """
    Inverse of the survival function of the slash distribution.

    This function numerically inverts the survival function so it
    may be slow, and in some cases it may fail to find a solution.
    """
    with mp.extradps(5):
        p = _validate_p(p)
        if p == 0:
            return mp.inf
        if p == 1:
            return mp.ninf
        if p == 0.5:
            return mp.zero
        if p > 0.5:
            x0 = -_npdf0/(1 - p)
        else:
            x0 = _npdf0/p
        return mp.findroot(lambda x: sf(x) - p, x0=x0)
