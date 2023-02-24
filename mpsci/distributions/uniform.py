"""
Uniform distribution
--------------------

These functions are for the uniform distribution on [a, b].
"""

from mpmath import mp
from ._common import _validate_p, _seq_to_mp
from ..stats import mean as _mean


__all__ = ['pdf', 'logpdf', 'cdf', 'sf', 'invcdf', 'invsf',
           'mean', 'var', 'median', 'entropy', 'mle', 'mom']


def _validate(a, b):
    if a >= b:
        raise ValueError('`a` must be less than `b`.')
    return mp.mpf(a), mp.mpf(b)


def pdf(x, a=0, b=1):
    """
    Uniform distribution probability density function.
    """
    with mp.extradps(5):
        a, b, = _validate(a, b)
        x = mp.mpf(x)
        if x < a or x > b:
            return mp.zero
        return mp.one / (b - a)


def logpdf(x, a=0, b=1):
    """
    Logarithm of the PDF of the uniform distribution.
    """
    with mp.extradps(5):
        a, b = _validate(a, b)
        x = mp.mpf(x)
        if x < a or x > b:
            return mp.ninf
        return -mp.log(b - a)


def cdf(x, a=0, b=1):
    """
    Uniform distribution cumulative distribution function.
    """
    with mp.extradps(5):
        a, b = _validate(a, b)
        x = mp.mpf(x)
        if x < a:
            return mp.zero
        elif x > b:
            return mp.one
        else:
            return (x - a) / (b - a)


def sf(x, a=0, b=1):
    """
    Uniform distribution survival function.
    """
    with mp.extradps(5):
        a, b = _validate(a, b)
        x = mp.mpf(x)
        if x < a:
            return mp.one
        elif x > b:
            return mp.zero
        else:
            return (b - x) / (b - a)


def invcdf(p, a=0, b=1):
    """
    Uniform distribution inverse CDF.

    This function is also known as the quantile function or the percent
    point function.
    """
    with mp.extradps(5):
        p = _validate_p(p)
        a, b = _validate(a, b)
        x = a + p*(b - a)
        return x


def invsf(p, a=0, b=1):
    """
    Unifiorm distribution inverse survival function.
    """
    with mp.extradps(5):
        p = _validate_p(p)
        a, b = _validate(a, b)
        x = b - p*(b - a)
        return x


def mean(a=0, b=1):
    """
    Mean of the uniform distribution.
    """
    with mp.extradps(5):
        a, b = _validate(a, b)
        return (a + b) / 2


def median(a=0, b=1):
    """
    Median of the uniform distribution.
    """
    with mp.extradps(5):
        a, b = _validate(a, b)
        return (a + b) / 2


def var(a=0, b=1):
    """
    Variance of the uniform distribution.
    """
    with mp.extradps(5):
        a, b = _validate(a, b)
        return (b - a)**2 / 12


def entropy(a=0, b=1):
    """
    Entropy of the uniform distribution.
    """
    with mp.extradps(5):
        a, b = _validate(a, b)
        return mp.log(b - a)


def mle(x):
    """
    Uniform distribution maximum likelihood parameter estimation.

    Returns (a, b).
    """
    with mp.extradps(5):
        x = _seq_to_mp(x)
        return min(x), max(x)


def mom(x):
    """
    Method of moments parameter estimation for the uniform distribution.

    Returns (a, b).
    """
    with mp.extradps(5):
        x = _seq_to_mp(x)
        M1 = _mean(x)
        M2 = _mean([t**2 for t in x])
        v = mp.sqrt(3*(M2 - M1**2))
        return M1 - v, M1 + v
