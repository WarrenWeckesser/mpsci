"""
Beta probability distribution
-----------------------------

"""
import mpmath
from ..fun import logbeta, xlogy, xlog1py


__all__ = ['pdf', 'logpdf', 'cdf', 'invcdf', 'sf', 'invsf', 'mean']


def pdf(x, a, b):
    """
    Probability density function (PDF) for the beta distribution.
    """
    if a <= 0 or b <= 0:
        raise ValueError('The shape parameters a and b must be greater '
                         'than 0.')
    x = mpmath.mpf(x)
    a = mpmath.mpf(a)
    b = mpmath.mpf(b)
    if x < 0 or x > 1:
        return mpmath.mp.zero
    if x == 0 and a < 1:
        return mpmath.inf
    if x == 1 and b < 1:
        return mpmath.inf
    return (mpmath.power(x, a - 1) * mpmath.power(1 - x, b - 1) /
            mpmath.beta(a, b))


def logpdf(x, a, b):
    """
    Natural logarithm of the PDF of the beta distribution.
    """
    if a <= 0 or b <= 0:
        raise ValueError('The shape parameters a and b must be greater '
                         'than 0.')
    x = mpmath.mpf(x)
    a = mpmath.mpf(a)
    b = mpmath.mpf(b)
    if x < 0 or x > 1:
        return -mpmath.mp.inf
    return xlogy(a - 1, x) + xlog1py(b - 1, -x) - logbeta(a, b)


def cdf(x, a, b):
    """
    Cumulative distribution function of the beta distribution.
    """
    if a <= 0 or b <= 0:
        raise ValueError('The shape parameters a and b must be greater '
                         'than 0.')
    if x < 0:
        return mpmath.mp.zero
    if x > 1:
        return mpmath.mp.one
    return mpmath.betainc(a, b, x1=0, x2=x, regularized=True)


def sf(x, a, b):
    """
    Survival function of the beta distribution.
    """
    if a <= 0 or b <= 0:
        raise ValueError('The shape parameters a and b must be greater '
                         'than 0.')
    return mpmath.betainc(a, b, x1=x, x2=1, regularized=True)


def invcdf(p, a, b):
    """
    Inverse of the CDF of the beta distribution.
    """
    if a <= 0 or b <= 0:
        raise ValueError('The shape parameters a and b must be greater '
                         'than 0.')
    if p < 0 or p > 1:
        return mpmath.nan
    if p == 0:
        return mpmath.mp.zero
    if p == 1:
        return mpmath.mp.one

    # XXX Bisection is not the most efficient method.  This also fails in some
    # cases when p is very close to 0 or 1.
    x = mpmath.findroot(lambda x: cdf(x, a, b) - p, x0=(0, 1), solver='bisect')
    return x


def invsf(p, a, b):
    """
    Inverse of the survival function of the beta distribution.
    """
    if a <= 0 or b <= 0:
        raise ValueError('The shape parameters a and b must be greater '
                         'than 0.')
    if p < 0 or p > 1:
        return mpmath.nan
    if p == 0:
        return mpmath.mp.one
    if p == 1:
        return mpmath.mp.zero

    # XXX Bisection is not the most efficient method.  This also fails in some
    # cases when p is very close to 0 or 1.
    x = mpmath.findroot(lambda x: sf(x, a, b) - p, x0=(0, 1), solver='bisect')
    return x


def mean(a, b):
    """
    Mean of the beta distribution.
    """
    if a <= 0 or b <= 0:
        raise ValueError('The shape parameters a and b must be greater '
                         'than 0.')
    a = mpmath.mpf(a)
    b = mpmath.mpf(b)
    return a/(a + b)