"""
Slash distribution
------------------

See https://en.wikipedia.org/wiki/Slash_distribution for details.

"""

import mpmath


__all__ = ['pdf', 'logpdf', 'cdf', 'sf']


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
        return mpmath.ncdf(x) - (mpmath.npdf(0) - mpmath.npdf(x))/x


def sf(x, mu=0, sigma=1):
    """
    Survival function for the slash distribution.
    """
    with mpmath.extradps(5):
        if x == 0:
            return mpmath.mp.one/2
        return mpmath.ncdf(-x) + (mpmath.npdf(0) - mpmath.npdf(x))/x
