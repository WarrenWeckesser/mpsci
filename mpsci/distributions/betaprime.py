"""
Beta prime probability distribution
-----------------------------------

See https://en.wikipedia.org/wiki/Beta_prime_distribution

"""
import mpmath
from ._common import _validate_p, _find_bracket
from .. import fun as _fun


__all__ = ['pdf', 'logpdf', 'cdf', 'invcdf', 'sf', 'invsf',
           'mean', 'mode', 'var', 'skewness']


def _validate_a_b(a, b):
    if a <= 0 or b <= 0:
        raise ValueError('The shape parameters a and b must be greater '
                         'than 0.')
    a = mpmath.mpf(a)
    b = mpmath.mpf(b)
    return a, b


def pdf(x, a, b):
    """
    Probability density function (PDF) for the beta prime distribution.
    """
    with mpmath.extradps(5):
        a, b = _validate_a_b(a, b)
        x = mpmath.mpf(x)
        if x < 0:
            return mpmath.mp.zero
        if x == 0 and a < 1:
            return mpmath.inf
        return (mpmath.power(x, a - 1) / mpmath.power(1 + x, a + b) /
                mpmath.beta(a, b))


def logpdf(x, a, b):
    """
    Natural logarithm of the PDF of the beta prime distribution.
    """
    with mpmath.extradps(5):
        a, b = _validate_a_b(a, b)
        x = mpmath.mpf(x)
        if x < 0:
            return -mpmath.mp.inf
        if x == 0 and a < 1:
            return mpmath.inf
        return (_fun.xlogy(a - 1, x) - _fun.xlog1py(a + b, x)
                - _fun.logbeta(a, b))


def cdf(x, a, b):
    """
    Cumulative distribution function of the beta prime distribution.
    """
    with mpmath.extradps(5):
        a, b = _validate_a_b(a, b)
        x = mpmath.mpf(x)
        if x < 0:
            return mpmath.mp.zero
        if x == mpmath.inf:
            return mpmath.mp.one
        if x > 1:
            c = mpmath.betainc(b, a, x1=1/(1+x), x2=1, regularized=True)
        else:
            c = mpmath.betainc(a, b, x1=0, x2=x/(1+x), regularized=True)
        return c


def sf(x, a, b):
    """
    Survival function of the beta prime distribution.
    """
    with mpmath.extradps(5):
        a, b = _validate_a_b(a, b)
        x = mpmath.mpf(x)
        if x < 0:
            return mpmath.mp.one
        if x == mpmath.inf:
            return mpmath.mp.zero
        if x > 1:
            c = mpmath.betainc(b, a, x1=0, x2=1/(1+x), regularized=True)
        else:
            c = mpmath.betainc(a, b, x1=x/(1+x), x2=1, regularized=True)
        return c


def invcdf(p, a, b):
    """
    Inverse of the CDF of the beta prime distribution.
    """
    with mpmath.extradps(5):
        a, b = _validate_a_b(a, b)
        p = _validate_p(p)
        if p == 0:
            return mpmath.mp.zero
        if p == 1:
            return mpmath.mp.inf

        x0, x1 = _find_bracket(lambda x: cdf(x, a, b), p, 0, mpmath.inf)
        if x0 == x1:
            return x0
        x = mpmath.findroot(lambda x: cdf(x, a, b) - p, x0=(x0, x1),
                            solver='secant')
        return x


def invsf(p, a, b):
    """
    Inverse of the survival function of the beta prime distribution.
    """
    with mpmath.extradps(5):
        a, b = _validate_a_b(a, b)
        p = _validate_p(p)
        if p == 0:
            return mpmath.mp.one
        if p == 1:
            return mpmath.mp.zero

        x0, x1 = _find_bracket(lambda x: sf(x, a, b), p, 0, mpmath.inf)
        if x0 == x1:
            return x0
        x = mpmath.findroot(lambda x: sf(x, a, b) - p, x0=(x0, x1),
                            solver='secant')
        return x


def mean(a, b):
    """
    Mean of the beta prime distribution.
    """
    with mpmath.extradps(5):
        a, b = _validate_a_b(a, b)
        if b <= 1:
            return mpmath.mp.inf
        return a / (b - 1)


def mode(a, b):
    """
    Mode of the beta prime distribution.
    """
    with mpmath.extradps(5):
        a, b = _validate_a_b(a, b)
        if a < 1:
            return mpmath.mp.zero
        return (a - 1) / (b + 1)


def var(a, b):
    """
    Variance of the beta prime distribution.
    """
    with mpmath.extradps(5):
        a, b = _validate_a_b(a, b)
        if b <= 2:
            return mpmath.mp.nan
        return (a * (a + b - 1)) / ((b - 2)*(b - 1)**2)


def skewness(a, b):
    """
    Skewness of the beta prime distribution.
    """
    with mpmath.extradps(5):
        a, b = _validate_a_b(a, b)
        if b <= 3:
            return mpmath.mp.nan
        t1 = 2 * (2*a + b - 1) / (b - 3)
        t2 = mpmath.sqrt((b - 2) / (a*(a + b - 1)))
        return t1 * t2
