"""
Chi distribution
----------------
"""

import re
from mpmath import mp
from ._common import _validate_moment_n


__all__ = ['pdf', 'logpdf', 'cdf', 'sf', 'mode', 'mean', 'var',
           'entropy', 'noncentral_moment']


def _validate_k(k):
    if k <= 0:
        raise ValueError('k must be positive')
    return mp.mpf(k)


def pdf(x, k):
    """
    PDF for the chi distribution.

    The PDF is

        f(x; k) = x**(k - 1) * exp(-x**2/2) / (2**(k/2 - 1) * Gamma(k/2))

    for x >= 0.
    """
    with mp.extradps(5):
        k = _validate_k(k)
        if x < 0:
            return mp.zero
        return mp.exp(logpdf(x, k))


_pdf_docstring_replace = r"""
    .. math::

       f(x; k) = \\frac{x^{k-1}e^{-x^2/2}}{2^{k/2-1}\\Gamma\\left(\\frac{k}{2}\\right)}

"""

pdf._docstring_re_subs = [
    (r'    f.*k/2\)\)', _pdf_docstring_replace, 0, re.DOTALL),
    (r'x >= 0', r':math:`x \\ge 0`', 0, 0),

]


def logpdf(x, k):
    """
    Logarithm of the PDF for the chi distribution.
    """
    with mp.extradps(5):
        k = _validate_k(k)
        x = mp.mpf(x)
        if x < 0:
            return mp.ninf
        p = ((k - 1)*mp.log(x) - x**2/2 - ((k/2) - 1)*mp.log(2)
             - mp.loggamma(k/2))
        return p


def cdf(x, k):
    """
    CDF for the chi distribution.
    """
    with mp.extradps(5):
        k = _validate_k(k)
        x = mp.mpf(x)
        if x <= 0:
            return mp.zero
        c = mp.gammainc(k/2, a=0, b=x**2/2, regularized=True)
        return c


def sf(x, k):
    """
    Survival function for the chi distribution.
    """

    with mp.extradps(5):
        k = _validate_k(k)
        x = mp.mpf(x)
        if x <= 0:
            return mp.one
        s = mp.gammainc(k/2, a=x**2/2, b=mp.inf, regularized=True)
    return s


def mode(k):
    """
    Mode of the chi distribution.

    The mode is sqrt(k - 1) for k >= 1.

    For 0 < k < 1, 0 is returned.
    """
    with mp.extradps(5):
        k = _validate_k(k)
        if k < 1:
            return mp.zero
        return mp.sqrt(k - 1)


mode._docstring_re_subs = ([
    (r'sqrt\(k - 1\)', r':math:`\\sqrt{k - 1}`', 0, 0),
    (r'k >= 1', r':math:`k \\ge 1`', 0, 0),
    (r'0 < k < 1', r':math:`0 < k < 1`', 0, 0),
])


def mean(k):
    """
    Mean of the chi distribution.
    """
    with mp.extradps(5):
        k = _validate_k(k)
        return mp.sqrt(2) * mp.gamma((k + 1)/2) / mp.gamma(k/2)
        return k


def var(k):
    """
    Variance of the chi distribution.
    """
    with mp.extradps(5):
        k = _validate_k(k)
        mu = mean(k)
        return k - mu**2


def entropy(k):
    """
    Differential entropy of the chi distribution.
    """
    with mp.extradps(5):
        k = _validate_k(k)
        k2 = k/2
        return mp.loggamma(k2) + (k - mp.log(2) - (k - 1)*mp.psi(0, k2))/2


def noncentral_moment(n, k):
    """
    n-th noncentral moment of the chi distribution.

    n must be a nonnegative integer.
    """
    n = _validate_moment_n(n)
    with mp.extradps(5):
        k = _validate_k(k)
        if n == 0:
            return mp.one
        return mp.mpf(2)**(n/2) * mp.gammaprod([(n + k)/2], [k/2])
