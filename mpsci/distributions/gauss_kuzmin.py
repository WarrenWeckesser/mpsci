"""
Gauss-Kuzmin distribution
-------------------------

The Gauss–Kuzmin distribution is a discrete probability distribution
that arises as the limit probability distribution of the coefficients
in the continued fraction expansion of a random variable uniformly
distributed in (0, 1).

See https://en.wikipedia.org/wiki/Gauss%E2%80%93Kuzmin_distribution

"""

from mpmath import mp
from ._common import _validate_p


__all__ = ['pmf', 'logpmf', 'cdf', 'invcdf', 'sf', 'invsf',
           'mode', 'median', 'mean', 'var']


def pmf(k):
    """
    Probability mass function (PMF) of the Gauss-Kuzmin distribution.

    `k` is expected be an integer; the code does not check this.
    """
    if k < 1:
        return mp.zero
    with mp.extradps(5):
        k = mp.mpf(k)
        r = 1/(k + 1)
        return -(mp.log1p(-r) + mp.log1p(r)) / mp.log(2)


def logpmf(k):
    """
    Logarithm of the PMF of the Gauss-Kuzmin distribution.

    `k` is expected be an integer; the code does not check this.
    """
    if k < 0:
        return mp.ninf
    with mp.extradps(5):
        k = mp.mpf(k)
        return mp.log(pmf(k))


def cdf(k):
    """
    CDF of the Gauss-Kuzmin distribution.

    `k` is expected be an integer; the code does not check this.
    """
    if k < 0:
        return mp.zero
    with mp.extradps(5):
        k = mp.mpf(k)
        if mp.isinf(k):
            return mp.one
        r = 1/(k + 1)
        return mp.one - mp.log1p(r)/mp.log(2)


def invcdf(p):
    """
    Inverse of the CDF of the Gauss-Kuzmin distribution.

    The distribution is discrete, but mpmath.mpf values are returned,
    to allow for returning `inf` when p is 1.
    """
    with mp.extradps(5):
        p = _validate_p(p)
        if p == 1:
            return mp.inf
        t = mp.powm1(2, 1 - p)
        return mp.powm1(t, -1)


def sf(k):
    """
    Survival function of the Gauss-Kuzmin distribution.

    `k` is expected be an integer; the code does not check this.
    """
    if k < 0:
        return mp.one
    with mp.extradps(5):
        k = mp.mpf(k)
        r = 1/(k + 1)
        return mp.log1p(r)/mp.log(2)


def invsf(p):
    """
    Inverse of the survival function of the Gauss-Kuzmin distribution.

    The distribution is discrete, but mpmath.mpf values are returned,
    to allow for returning `inf` when p is 0.
    """
    with mp.extradps(5):
        p = _validate_p(p)
        if p == 0:
            return mp.inf
        t = mp.powm1(2, p)
        return mp.powm1(t, -1)


def mode():
    """
    Mode of the Gauss-Kuzmin distribution.
    """
    return mp.one


def median():
    """
    Median of the Gauss-Kuzmin distribution.
    """
    return mp.mpf(2)


def mean():
    """
    Mean of the Gauss-Kuzmin distribution.
    """
    return mp.inf


def var():
    """
    Variance of the Gauss-Kuzmin distribution.
    """
    return mp.inf
