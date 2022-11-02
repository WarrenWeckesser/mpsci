"""
Gauss-Kuzmin distribution
-------------------------

The Gauss–Kuzmin distribution is a discrete probability distribution
that arises as the limit probability distribution of the coefficients
in the continued fraction expansion of a random variable uniformly
distributed in (0, 1).

See https://en.wikipedia.org/wiki/Gauss%E2%80%93Kuzmin_distribution

"""

import mpmath
from ._common import _validate_p


__all__ = ['pmf', 'logpmf', 'cdf', 'invcdf', 'sf', 'invsf',
           'mode', 'median', 'mean', 'var']


def pmf(k):
    """
    Probability mass function (PMF) of the Gauss-Kuzmin distribution.

    `k` is expected be an integer; the code does not check this.
    """
    if k < 1:
        return mpmath.mp.zero
    with mpmath.extradps(5):
        k = mpmath.mpf(k)
        r = 1/(k + 1)
        return -(mpmath.log1p(-r) + mpmath.log1p(r)) / mpmath.log(2)


def logpmf(k):
    """
    Logarithm of the PMF of the Gauss-Kuzmin distribution.

    `k` is expected be an integer; the code does not check this.
    """
    if k < 0:
        return mpmath.ninf
    with mpmath.extradps(5):
        k = mpmath.mpf(k)
        return mpmath.log(pmf(k))


def cdf(k):
    """
    CDF of the Gauss-Kuzmin distribution.

    `k` is expected be an integer; the code does not check this.
    """
    if k < 0:
        return mpmath.mp.zero
    with mpmath.extradps(5):
        k = mpmath.mpf(k)
        if mpmath.isinf(k):
            return mpmath.mp.one
        r = 1/(k + 1)
        return mpmath.mp.one - mpmath.log1p(r)/mpmath.log(2)


def invcdf(p):
    """
    Inverse of the CDF of the Gauss-Kuzmin distribution.

    The distribution is discrete, but mpmath.mpf values are returned,
    to allow for returning `inf` when p is 1.
    """
    with mpmath.extradps(5):
        p = _validate_p(p)
        if p == 1:
            return mpmath.inf
        t = mpmath.powm1(2, 1 - p)
        return mpmath.powm1(t, -1)


def sf(k):
    """
    Survival function of the Gauss-Kuzmin distribution.

    `k` is expected be an integer; the code does not check this.
    """
    if k < 0:
        return mpmath.mp.one
    with mpmath.extradps(5):
        k = mpmath.mpf(k)
        r = 1/(k + 1)
        return mpmath.log1p(r)/mpmath.log(2)


def invsf(p):
    """
    Inverse of the survival function of the Gauss-Kuzmin distribution.

    The distribution is discrete, but mpmath.mpf values are returned,
    to allow for returning `inf` when p is 0.
    """
    with mpmath.extradps(5):
        p = _validate_p(p)
        if p == 0:
            return mpmath.inf
        t = mpmath.powm1(2, p)
        return mpmath.powm1(t, -1)


def mode():
    """
    Mode of the Gauss-Kuzmin distribution.
    """
    return mpmath.mp.one


def median():
    """
    Median of the Gauss-Kuzmin distribution.
    """
    return mpmath.mpf(2)


def mean():
    """
    Mean of the Gauss-Kuzmin distribution.
    """
    return mpmath.inf


def var():
    """
    Variance of the Gauss-Kuzmin distribution.
    """
    return mpmath.inf
