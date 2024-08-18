"""
Log-series distribution
-----------------------

This discrete distributions is also known as the logarithmic
distribution [1]_.

.. [1] Logarithmic distribution,
       https://en.wikipedia.org/wiki/Logarithmic_distribution
"""

import itertools
from mpmath import mp
from ._common import _validate_p


__all__ = ['support', 'pmf', 'logpmf', 'cdf', 'sf', 'mean', 'var', 'mode',
           'skewness', 'kurtosis']


def support(p):
    """
    Support of the log-series distribution.

    The support is the integers 1, 2, 3, ..., so the support is returned
    as an instance of `itertools.count(start=1)`.

    Examples
    --------
    >>> from mpsci.distributions import logseries
    >>> sup = logseries.support()
    >>> next(sup)
    1
    >>> next(sup)
    2

    """
    return itertools.count(start=1)


def pmf(k, p):
    """
    Probability mass function of the log-series distribution.
    """
    with mp.extradps(5):
        p = _validate_p(p)
        if k < 1:
            return mp.zero
        return mp.exp(logpmf(k, p))


def logpmf(k, p):
    """
    Natural log of the PMF of the log-series distribution.
    """
    with mp.extradps(5):
        p = _validate_p(p)
        if k < 1:
            return mp.ninf
        return k*mp.log(p) - mp.log(k) - mp.log(-mp.log1p(-p))


def cdf(k, p):
    """
    CDF of the log-series distribution.
    """
    with mp.extradps(5):
        p = _validate_p(p)
        if k < 1:
            return mp.zero
        return 1 + mp.betainc(k + 1, 0, 0, p) / mp.log1p(-p)


def sf(k, p):
    """
    Survival function of the log-series distribution.
    """
    with mp.extradps(5):
        p = _validate_p(p)
        if k < 1:
            return mp.one
        return -mp.betainc(k + 1, 0, 0, p) / mp.log1p(-p)


def mean(p):
    """
    Mean of the log-series distribution.
    """
    with mp.extradps(5):
        p = _validate_p(p)
        return p / (p - 1) / mp.log1p(-p)


def var(p):
    """
    Variance of the log-series distribution.
    """
    with mp.extradps(5):
        p = _validate_p(p)
        l1p = mp.log1p(-p)
        return -(p*(p + l1p)) / (1 - p)**2 / l1p**2


def mode(p):
    """
    Mode of the log-series distribution.
    """
    p = _validate_p(p)
    return mp.one


def skewness(p):
    """
    Skewness of the log-series distribution.
    """
    with mp.extradps(5):
        p = _validate_p(p)
        r = mp.log1p(-p)
        s = p + r
        num = p*(2*p + 3*r) + (1 + p)*r**2
        den = -mp.sqrt(-p*s)*s
        return num/den


def kurtosis(p):
    """
    Excess kurtosis of the log-series distribution.
    """
    with mp.extradps(5):
        p = _validate_p(p)
        r = mp.log1p(-p)
        r2 = r*r
        r3 = r2*r
        num = p*(p*(-6*p + r*(r*(-r - 4) - 12)) + r2*(-4*r - 7)) - r3
        den = p*(p + r)**2
        return num/den
