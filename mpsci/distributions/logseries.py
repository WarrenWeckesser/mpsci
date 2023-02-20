"""
Log-series distribution
-----------------------

This discrete distributions is also known as the logarithmic
distribution [1]_.

.. [1] Logarithmic distribution,
       https://en.wikipedia.org/wiki/Logarithmic_distribution
"""

from mpmath import mp
from ._common import _validate_p


__all__ = ['pmf', 'logpmf', 'cdf', 'sf', 'mean', 'var', 'mode']


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
