"""
Log-series distribution
-----------------------

This discrete distributions is also known as the logarithmic
distribution [1]_.

.. [1] Logarithmic distribution,
       https://en.wikipedia.org/wiki/Logarithmic_distribution
"""

import mpmath as _mpm
from ._common import _validate_p


__all__ = ['pmf', 'logpmf', 'cdf', 'sf', 'mean', 'var', 'mode']


def pmf(k, p):
    """
    Probability mass function of the log-series distribution.
    """
    with _mpm.extradps(5):
        p = _validate_p(p)
        if k < 1:
            return _mpm.mp.zero
        return _mpm.exp(logpmf(k, p))


def logpmf(k, p):
    """
    Natural log of the PMF of the log-series distribution.
    """
    with _mpm.extradps(5):
        p = _validate_p(p)
        if k < 1:
            return -_mpm.mp.inf
        return k*_mpm.log(p) - _mpm.log(k) - _mpm.log(-_mpm.log1p(-p))


def cdf(k, p):
    """
    CDF of the log-series distribution.
    """
    with _mpm.extradps(5):
        p = _validate_p(p)
        if k < 1:
            return _mpm.mp.zero
        return 1 + _mpm.betainc(k + 1, 0, 0, p) / _mpm.log1p(-p)


def sf(k, p):
    """
    Survival function of the log-series distribution.
    """
    with _mpm.extradps(5):
        p = _validate_p(p)
        if k < 1:
            return _mpm.mp.one
        return -_mpm.betainc(k + 1, 0, 0, p) / _mpm.log1p(-p)


def mean(p):
    """
    Mean of the log-series distribution.
    """
    with _mpm.extradps(5):
        p = _validate_p(p)
        return p / (p - 1) / _mpm.log1p(-p)


def var(p):
    """
    Variance of the log-series distribution.
    """
    with _mpm.extradps(5):
        p = _validate_p(p)
        l1p = _mpm.log1p(-p)
        return -(p*(p + l1p)) / (1 - p)**2 / l1p**2


def mode(p):
    """
    Mode of the log-series distribution.
    """
    p = _validate_p(p)
    return _mpm.mp.one
