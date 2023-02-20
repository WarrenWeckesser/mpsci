"""
Exponentiated Weibull probability distribution
----------------------------------------------
"""

from mpmath import mp


__all__ = ['pdf', 'logpdf', 'cdf']


def pdf(x, a, c, scale=1):
    """
    PDF for the exponentiated Weibull distribution.

    All the distribution parameters are assumed to be positive.
    """
    x = mp.mpf(x)
    a = mp.mpf(a)
    c = mp.mpf(c)
    scale = mp.mpf(scale)

    if x < 0:
        return mp.zero

    z = x / scale
    p = (a * c / scale * z**(c-1) * (-mp.expm1(-z**c))**(a - 1) *
         mp.exp(-z**c))
    return p


def logpdf(x, a, c, scale=1):
    """
    Logarithm of the PDF for the exponentiated Weibull distribution.

    All the distribution parameters are assumed to be positive.
    """
    x = mp.mpf(x)
    a = mp.mpf(a)
    c = mp.mpf(c)
    scale = mp.mpf(scale)

    if x < 0:
        return mp.ninf

    z = x / scale
    logp = (mp.log(a)
            + mp.log(c)
            - mp.log(scale)
            + (c - 1)*mp.log(z)
            + (a - 1)*mp.log(-mp.expm1(-z**c))
            - z**c)
    return logp


def cdf(x, a, c, scale=1):
    """
    CDF for the exponentiated Weibull distribution.

    This is the cumulative distribution function for the exponentiated
    Weibull distribution.

    All the distribution parameters are assumed to be positive.
    """
    x = mp.mpf(x)
    a = mp.mpf(a)
    c = mp.mpf(c)
    scale = mp.mpf(scale)

    if x < 0:
        return mp.zero

    z = x / scale
    return mp.power(-mp.expm1(-z**c), a)
