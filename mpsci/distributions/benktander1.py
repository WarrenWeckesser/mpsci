"""
Benktander I Distribution
-------------------------
"""

import mpmath


__all__ = ['pdf', 'logpdf', 'cdf', 'sf', 'mean']


def pdf(x, a, b):
    """
    PDF of the Benktander I distribution.

    Variable names follow the convention used on wikipedia.
    """
    if a <= 0:
        raise ValueError("'a' must be positive.")
    if b <= 0:
        raise ValueError("'b' must be positive.")
    if x < 1:
        return mpmath.mp.zero
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        a = mpmath.mpf(a)
        b = mpmath.mpf(b)
        blogx = b*mpmath.log(x)
        c = (1 + 2*blogx/a)*(1 + 2 + 2*blogx) - 2*b/a
        return c * mpmath.power(x, -(2 + a + blogx))


def logpdf(x, a, b):
    """
    Logarithm of the PDF of the Benktander I distribution.

    Variable names follow the convention used on wikipedia.
    """
    if a <= 0:
        raise ValueError("'a' must be positive.")
    if b <= 0:
        raise ValueError("'b' must be positive.")
    if x < 1:
        return mpmath.ninf
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        a = mpmath.mpf(a)
        b = mpmath.mpf(b)
        blogx = b*mpmath.log(x)
        c = (1 + 2*blogx/a)*(1 + 2 + 2*blogx) - 2*b/a
        return mpmath.log(c) - (2 + a + blogx)*mpmath.log(x)


def cdf(x, a, b):
    """
    CDF of the Benktander I distribution.

    Variable names follow the convention used on wikipedia.
    """
    if a <= 0:
        raise ValueError("'a' must be positive.")
    if b <= 0:
        raise ValueError("'b' must be positive.")
    if x < 1:
        return mpmath.mp.zero
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        a = mpmath.mpf(a)
        b = mpmath.mpf(b)
        blogx = b*mpmath.log(x)
        return 1 - (1 + 2*blogx/a)*mpmath.power(x, -(a + 1 + blogx))


def sf(x, a, b):
    """
    Survival function of the Benktander I distribution.

    Variable names follow the convention used on wikipedia.
    """
    if a <= 0:
        raise ValueError("'a' must be positive.")
    if b <= 0:
        raise ValueError("'b' must be positive.")
    if x < 1:
        return mpmath.mp.one
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        a = mpmath.mpf(a)
        b = mpmath.mpf(b)
        blogx = b*mpmath.log(x)
        return (1 + 2*blogx/a)*mpmath.power(x, -(a + 1 + blogx))


def mean(a, b):
    """
    Mean of the Benktander I distribution.

    Variable names follow the convention used on wikipedia.
    """
    if a <= 0:
        raise ValueError("'a' must be positive.")
    if b <= 0:
        raise ValueError("'b' must be positive.")
    with mpmath.extradps(5):
        return 1 + 1/a
