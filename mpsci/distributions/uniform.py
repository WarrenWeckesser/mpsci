"""
Uniform distribution
--------------------

These functions are for the uniform distribution on [a, b].
"""

import mpmath as _mpmath
from ..stats import mean as _mean


__all__ = ['pdf', 'logpdf', 'cdf', 'sf', 'invcdf', 'invsf',
           'mean', 'var', 'median', 'entropy', 'mle', 'mom']


def _validate(a, b):
    if a >= b:
        raise ValueError('`a` must be less than `b`.')
    return _mpmath.mpf(a), _mpmath.mpf(b)


def pdf(x, a=0, b=1):
    """
    Uniform distribution probability density function.
    """
    with _mpmath.extradps(5):
        a, b, = _validate(a, b)
        x = _mpmath.mpf(x)
        if x < a or x > b:
            return _mpmath.mp.zero
        return _mpmath.mp.one / (b - a)


def logpdf(x, a=0, b=1):
    """
    Logarithm of the PDF of the uniform distribution.
    """
    with _mpmath.extradps(5):
        a, b = _validate(a, b)
        x = _mpmath.mpf(x)
        if x < a or x > b:
            return _mpmath.mp.ninf
        return -_mpmath.log(b - a)


def cdf(x, a=0, b=1):
    """
    Uniform distribution cumulative distribution function.
    """
    with _mpmath.extradps(5):
        a, b = _validate(a, b)
        x = _mpmath.mpf(x)
        if x < a:
            return _mpmath.mp.zero
        elif x > b:
            return _mpmath.mp.one
        else:
            return (x - a) / (b - a)


def sf(x, a=0, b=1):
    """
    Uniform distribution survival function.
    """
    with _mpmath.extradps(5):
        a, b = _validate(a, b)
        x = _mpmath.mpf(x)
        if x < a:
            return _mpmath.mp.one
        elif x > b:
            return _mpmath.mp.zero
        else:
            return (b - x) / (b - a)


def invcdf(p, a=0, b=1):
    """
    Uniform distribution inverse CDF.

    This function is also known as the quantile function or the percent
    point function.
    """
    with _mpmath.extradps(5):
        a, b = _validate(a, b)
        if p < 0 or p > 1:
            return _mpmath.nan
        p = _mpmath.mpf(p)
        x = a + p*(b - a)
        return x


def invsf(p, a=0, b=1):
    """
    Unifiorm distribution inverse survival function.
    """
    with _mpmath.extradps(5):
        a, b = _validate(a, b)
        if p < 0 or p > 1:
            return _mpmath.nan
        p = _mpmath.mpf(p)
        x = b - p*(b - a)
        return x


def mean(a=0, b=1):
    """
    Mean of the uniform distribution.
    """
    with _mpmath.extradps(5):
        a, b = _validate(a, b)
        return (a + b) / 2


def median(a=0, b=1):
    """
    Median of the uniform distribution.
    """
    with _mpmath.extradps(5):
        a, b = _validate(a, b)
        return (a + b) / 2


def var(a=0, b=1):
    """
    Variance of the uniform distribution.
    """
    with _mpmath.extradps(5):
        a, b = _validate(a, b)
        return (b - a)**2 / 12


def entropy(a=0, b=1):
    """
    Entropy of the uniform distribution.
    """
    with _mpmath.extradps(5):
        a, b = _validate(a, b)
        return _mpmath.log(b - a)


def mle(x):
    """
    Uniform distribution maximum likelihood parameter estimation.

    Returns (a, b).
    """
    with _mpmath.extradps(5):
        x = [_mpmath.mpf(t) for t in x]
        return min(x), max(x)


def mom(x):
    """
    Method of moments parameter estimation for the uniform distribution.

    Returns (a, b).
    """
    with _mpmath.extradps(5):
        M1 = _mean(x)
        M2 = _mean([_mpmath.mpf(t)**2 for t in x])
        v = _mpmath.sqrt(3*(M2 - M1**2))
        return M1 - v, M1 + v
