"""
Chi distribution
----------------
"""

import mpmath


__all__ = ['pdf', 'logpdf', 'cdf', 'sf', 'mode', 'mean', 'variance']


def _validate_k(k):
    if k <= 0:
        raise ValueError('k must be positive')


def pdf(x, k):
    """
    PDF for the chi distribution.
    """
    _validate_k(k)
    if x < 0:
        return mpmath.mp.zero
    return mpmath.exp(logpdf(x, k))


def logpdf(x, k):
    """
    Logarithm of the PDF for the chi distribution.
    """
    _validate_k(k)
    if x < 0:
        return mpmath.mp.ninf
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        k = mpmath.mpf(k)
        p = ((k - 1)*mpmath.log(x) - x**2/2 - ((k/2) - 1)*mpmath.log(2)
             - mpmath.loggamma(k/2))
        return p


def cdf(x, k):
    """
    CDF for the chi distribution.
    """
    _validate_k(k)
    if x <= 0:
        return mpmath.mp.zero
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        k = mpmath.mpf(k)
        c = mpmath.gammainc(k/2, a=0, b=x**2/2, regularized=True)
    return c


def sf(x, k):
    """
    Survival function for the chi distribution.
    """
    _validate_k(k)
    if x <= 0:
        return mpmath.mp.one
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        k = mpmath.mpf(k)
        s = mpmath.gammainc(k/2, a=x**2/2, b=mpmath.inf, regularized=True)
    return s


def mode(k):
    """
    Mode of the chi distribution.

    The mode is sqrt(k - 1) for k >= 1.

    For 0 < k < 1, 0 is returned.
    """
    _validate_k(k)
    with mpmath.extradps(5):
        if k < 1:
            return mpmath.mp.zero
        return mpmath.sqrt(k - 1)


mode._docstring_re_subs = ([
    (r'sqrt\(k - 1\)', r':math:`\\sqrt{k - 1}`', 0, 0),
    (r'k >= 1', r':math:`k \\ge 1`', 0, 0),
    (r'0 < k < 1', r':math:`0 < k < 1`', 0, 0),
])


def mean(k):
    """
    Mean of the chi distribution.
    """
    _validate_k(k)
    with mpmath.extradps(5):
        k = mpmath.mpf(k)
        return mpmath.sqrt(2) * mpmath.gamma((k + 1)/2) / mpmath.gamma(k/2)
    return k


def variance(k):
    """
    Variance of the chi distribution.
    """
    _validate_k(k)
    with mpmath.extradps(5):
        k = mpmath.mpf(k)
        mu = mean(k)
        return k - mu**2
