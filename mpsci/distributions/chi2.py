"""
Chi-square distribution
-----------------------
"""

import mpmath


__all__ = ['pdf', 'logpdf', 'cdf', 'sf', 'mean', 'mode', 'variance']


def _validate_k(k):
    if k <= 0:
        raise ValueError('k must be positive')


def pdf(x, k):
    """
    PDF for the chi-square distribution.
    """
    _validate_k(k)
    if x < 0:
        return mpmath.mp.zero
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        k = mpmath.mpf(k)
        p = mpmath.exp(-x/2) * (x/2)**(k/2 - 1)/2 / mpmath.gamma(k/2)
    return p


def logpdf(x, k):
    """
    Logarithm of the PDF for the chi-square distribution.
    """
    _validate_k(k)
    if x < 0:
        return mpmath.mp.zero
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        k = mpmath.mpf(k)
        p = (-x/2 + (k/2 - 1)*mpmath.log(x/2) - mpmath.log(2)
             - mpmath.loggamma(k/2))
    return p


def cdf(x, k):
    """
    CDF for the chi-square distribution.
    """
    _validate_k(k)
    if x <= 0:
        return mpmath.mp.zero
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        k = mpmath.mpf(k)
        c = mpmath.gammainc(k/2, a=0, b=x/2, regularized=True)
    return c


def sf(x, k):
    """
    Survival function for the chi-square distribution.
    """
    _validate_k(k)
    if x <= 0:
        return mpmath.mp.one
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        k = mpmath.mpf(k)
        s = mpmath.gammainc(k/2, a=x/2, b=mpmath.inf, regularized=True)
    return s


def mean(k):
    """
    Mean of the chi-square distribution.
    """
    _validate_k(k)
    k = mpmath.mpf(k)
    return k


def mode(k):
    """
    Mode of the chi-square distribution.

    The mode is max(k - 2, 0).
    """
    _validate_k(k)
    k = mpmath.mpf(k)
    return max(k - 2, mpmath.mp.zero)


def variance(k):
    """
    Variance of the chi-square distribution.
    """
    _validate_k(k)
    k = mpmath.mpf(k)
    return 2*k
