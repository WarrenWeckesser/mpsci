"""
Exponentiated Weibull probability distribution
----------------------------------------------
"""

from mpmath import mp
from ._common import _validate_p

__all__ = ['pdf', 'logpdf', 'cdf', 'invcdf', 'sf', 'invsf',
           'support']


def _validate_params(a, c, scale):
    if a <= 0:
        raise ValueError('`a` must be greater than 0')
    if c <= 0:
        raise ValueError('`c` must be greater than 0')
    if scale <= 0:
        raise ValueError('`scale` must be greater than 0')
    return mp.mpf(a), mp.mpf(c), mp.mpf(scale)


def pdf(x, a, c, scale=1):
    """
    PDF for the exponentiated Weibull distribution.

    All the distribution parameters are assumed to be positive.
    """
    with mp.extradps(5):
        a, c, scale = _validate_params(a, c, scale)
        x = mp.mpf(x)
        if x < 0:
            return mp.zero
        z = x/scale
        p = (a * c / scale * z**(c-1) * (-mp.expm1(-z**c))**(a - 1) *
             mp.exp(-z**c))
        return p


def logpdf(x, a, c, scale=1):
    """
    Logarithm of the PDF for the exponentiated Weibull distribution.

    All the distribution parameters are assumed to be positive.
    """
    with mp.extradps(5):
        a, c, scale = _validate_params(a, c, scale)
        x = mp.mpf(x)
        if x < 0:
            return mp.ninf
        z = x/scale
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
    with mp.extradps(5):
        a, c, scale = _validate_params(a, c, scale)
        x = mp.mpf(x)
        if x < 0:
            return mp.zero
        z = x/scale
        return mp.power(-mp.expm1(-z**c), a)


def invcdf(p, a, c, scale=1):
    """
    Inverse of the CDF of the exponentiated Weibull distribution.
    """
    with mp.extradps(5):
        a, c, scale = _validate_params(a, c, scale)
        p = _validate_p(p)
        return scale*(-mp.log1p(-p**(1/a)))**(1/c)


def sf(x, a, c, scale=1):
    """
    Survival function of the exponentiated Weibull distribution.
    """
    with mp.extradps(5):
        a, c, scale = _validate_params(a, c, scale)
        x = mp.mpf(x)
        if x < 0:
            return mp.one
        z = x/scale
        return -mp.powm1(-mp.expm1(-z**c), a)


def invsf(p, a, c, scale=1):
    """
    Inverse of the survival function of the exponentiated Weibull distribution.
    """
    with mp.extradps(5):
        a, c, scale = _validate_params(a, c, scale)
        p = _validate_p(p)
        return scale*(-mp.log(-mp.expm1(mp.log1p(-p)/a)))**(1/c)


def support(a, c, scale=1):
    """
    Support of the exponentiated Weibull distribution.
    """
    with mp.extradps(5):
        a, c, scale = _validate_params(a, c, scale)
        return (mp.zero, mp.inf)
