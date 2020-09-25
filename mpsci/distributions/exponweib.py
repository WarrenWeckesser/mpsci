"""
Exponentiated Weibull probability distribution
----------------------------------------------
"""

import mpmath


__all__ = ['pdf', 'logpdf', 'cdf']


def pdf(x, a, c, scale=1):
    """
    PDF for the exponentiated Weibull distribution.

    All the distribution parameters are assumed to be positive.
    """
    x = mpmath.mpf(x)
    a = mpmath.mpf(a)
    c = mpmath.mpf(c)
    scale = mpmath.mpf(scale)

    if x < 0:
        return mpmath.mp.zero

    z = x / scale
    p = (a * c / scale * z**(c-1) * (-mpmath.expm1(-z**c))**(a - 1) *
         mpmath.exp(-z**c))
    return p


def logpdf(x, a, c, scale=1):
    """
    Logarithm of the PDF for the exponentiated Weibull distribution.

    All the distribution parameters are assumed to be positive.
    """
    x = mpmath.mpf(x)
    a = mpmath.mpf(a)
    c = mpmath.mpf(c)
    scale = mpmath.mpf(scale)

    if x < 0:
        return -mpmath.mp.inf

    z = x / scale
    logp = (mpmath.log(a)
            + mpmath.log(c)
            - mpmath.log(scale)
            + (c - 1)*mpmath.log(z)
            + (a - 1)*mpmath.log(-mpmath.expm1(-z**c))
            - z**c)
    return logp


def cdf(x, a, c, scale=1):
    """
    CDF for the exponentiated Weibull distribution.

    This is the cumulative distribution function for the exponentiated
    Weibull distribution.

    All the distribution parameters are assumed to be positive.
    """
    x = mpmath.mpf(x)
    a = mpmath.mpf(a)
    c = mpmath.mpf(c)
    scale = mpmath.mpf(scale)

    if x < 0:
        return mpmath.mp.zero

    z = x / scale
    return mpmath.power(-mpmath.expm1(-z**c), a)
