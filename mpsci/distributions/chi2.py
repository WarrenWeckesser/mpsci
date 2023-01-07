"""
Chi-square distribution
-----------------------
"""

from mpmath import mp


__all__ = ['pdf', 'logpdf', 'cdf', 'sf', 'mean', 'mode', 'variance']


def _validate_k(k):
    if k <= 0:
        raise ValueError('k must be positive')
    return mp.mpf(k)


def pdf(x, k):
    """
    PDF for the chi-square distribution.
    """
    with mp.extradps(5):
        k = _validate_k(k)
        x = mp.mpf(x)
        if x < 0:
            return mp.zero
        p = mp.exp(-x/2) * (x/2)**(k/2 - 1)/2 / mp.gamma(k/2)
        return p


def logpdf(x, k):
    """
    Logarithm of the PDF for the chi-square distribution.
    """
    with mp.extradps(5):
        k = _validate_k(k)
        x = mp.mpf(x)
        if x < 0:
            return mp.ninf
        p = -x/2 + (k/2 - 1)*mp.log(x/2) - mp.log(2) - mp.loggamma(k/2)
    return p


def cdf(x, k):
    """
    CDF for the chi-square distribution.
    """
    with mp.extradps(5):
        k = _validate_k(k)
        x = mp.mpf(x)
        if x <= 0:
            return mp.zero
        c = mp.gammainc(k/2, a=0, b=x/2, regularized=True)
    return c


def sf(x, k):
    """
    Survival function for the chi-square distribution.
    """
    with mp.extradps(5):
        k = _validate_k(k)
        x = mp.mpf(x)
        if x <= 0:
            return mp.one
        s = mp.gammainc(k/2, a=x/2, b=mp.inf, regularized=True)
    return s


def mean(k):
    """
    Mean of the chi-square distribution.
    """
    with mp.extradps(5):
        k = _validate_k(k)
        return k


def mode(k):
    """
    Mode of the chi-square distribution.

    The mode is max(k - 2, 0).
    """
    with mp.extradps(5):
        k = _validate_k(k)
        return max(k - 2, mp.zero)


def variance(k):
    """
    Variance of the chi-square distribution.
    """
    with mp.extradps(5):
        k = _validate_k(k)
        return 2*k
