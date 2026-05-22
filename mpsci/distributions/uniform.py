"""
Uniform distribution
--------------------

These functions are for the uniform distribution on [a, b].
"""

from mpmath import mp
from ._common import _validate_p, _seq_to_mp, _validate_moment_n
from ..stats import mean as _mean


__all__ = ['pdf', 'logpdf', 'cdf', 'sf', 'invcdf', 'invsf',
           'support',
           'mean', 'var', 'median', 'entropy', 'noncentral_moment',
           'mle', 'mom']


def _validate(a, b):
    if a >= b:
        raise ValueError('`a` must be less than `b`.')
    return mp.mpf(a), mp.mpf(b)


@mp.extradps(5)
def pdf(x, a=0, b=1):
    """
    Uniform distribution probability density function.
    """
    a, b, = _validate(a, b)
    x = mp.mpf(x)
    if x < a or x > b:
        return mp.zero
    return mp.one / (b - a)


@mp.extradps(5)
def logpdf(x, a=0, b=1):
    """
    Logarithm of the PDF of the uniform distribution.
    """
    a, b = _validate(a, b)
    x = mp.mpf(x)
    if x < a or x > b:
        return mp.ninf
    return -mp.log(b - a)


@mp.extradps(5)
def cdf(x, a=0, b=1):
    """
    Uniform distribution cumulative distribution function.
    """
    a, b = _validate(a, b)
    x = mp.mpf(x)
    if x < a:
        return mp.zero
    elif x > b:
        return mp.one
    else:
        return (x - a) / (b - a)


@mp.extradps(5)
def sf(x, a=0, b=1):
    """
    Uniform distribution survival function.
    """
    a, b = _validate(a, b)
    x = mp.mpf(x)
    if x < a:
        return mp.one
    elif x > b:
        return mp.zero
    else:
        return (b - x) / (b - a)


@mp.extradps(5)
def invcdf(p, a=0, b=1):
    """
    Uniform distribution inverse CDF.

    This function is also known as the quantile function or the percent
    point function.
    """
    p = _validate_p(p)
    a, b = _validate(a, b)
    x = a + p*(b - a)
    return x


@mp.extradps(5)
def invsf(p, a=0, b=1):
    """
    Unifiorm distribution inverse survival function.
    """
    p = _validate_p(p)
    a, b = _validate(a, b)
    x = b - p*(b - a)
    return x


@mp.extradps(5)
def support(a=0, b=1):
    """
    Support of the uniform distribution.
    """
    a, b = _validate(a, b)
    return (a, b)


@mp.extradps(5)
def mean(a=0, b=1):
    """
    Mean of the uniform distribution.
    """
    a, b = _validate(a, b)
    return (a + b) / 2


@mp.extradps(5)
def median(a=0, b=1):
    """
    Median of the uniform distribution.
    """
    a, b = _validate(a, b)
    return (a + b) / 2


@mp.extradps(5)
def var(a=0, b=1):
    """
    Variance of the uniform distribution.
    """
    a, b = _validate(a, b)
    return (b - a)**2 / 12


@mp.extradps(5)
def entropy(a=0, b=1):
    """
    Entropy of the uniform distribution.
    """
    a, b = _validate(a, b)
    return mp.log(b - a)


@mp.extradps(5)
def noncentral_moment(n, a=0, b=1):
    """
    n-th noncentral moment of the uniform distribution.

    n must be a nonnegative integer.
    """
    n = _validate_moment_n(n)
    a, b = _validate(a, b)
    if n == 0:
        return mp.one
    return mp.fsum([a**i*b**(n - i) for i in range(n + 1)])/(n + 1)


@mp.extradps(5)
def mle(x):
    """
    Uniform distribution maximum likelihood parameter estimation.

    Returns (a, b).
    """
    x = _seq_to_mp(x)
    return min(x), max(x)


@mp.extradps(5)
def mom(x):
    """
    Method of moments parameter estimation for the uniform distribution.

    Returns (a, b).
    """
    x = _seq_to_mp(x)
    M1 = _mean(x)
    M2 = _mean([t**2 for t in x])
    v = mp.sqrt(3*(M2 - M1**2))
    return M1 - v, M1 + v
