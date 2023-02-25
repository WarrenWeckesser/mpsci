"""
Beta prime probability distribution
-----------------------------------

See https://en.wikipedia.org/wiki/Beta_prime_distribution

"""
from mpmath import mp
from ._common import _validate_p, _validate_moment_n, _find_bracket
from .. import fun as _fun


__all__ = ['pdf', 'logpdf', 'cdf', 'invcdf', 'sf', 'invsf',
           'mean', 'mode', 'var', 'skewness',
           'noncentral_moment']


def _validate_a_b(a, b):
    if a <= 0 or b <= 0:
        raise ValueError('The shape parameters a and b must be greater '
                         'than 0.')
    a = mp.mpf(a)
    b = mp.mpf(b)
    return a, b


def pdf(x, a, b):
    """
    Probability density function (PDF) for the beta prime distribution.
    """
    with mp.extradps(5):
        a, b = _validate_a_b(a, b)
        x = mp.mpf(x)
        if x < 0:
            return mp.zero
        if x == 0 and a < 1:
            return mp.inf
        return (mp.power(x, a - 1) / mp.power(1 + x, a + b) /
                mp.beta(a, b))


def logpdf(x, a, b):
    """
    Natural logarithm of the PDF of the beta prime distribution.
    """
    with mp.extradps(5):
        a, b = _validate_a_b(a, b)
        x = mp.mpf(x)
        if x < 0:
            return mp.ninf
        if x == 0 and a < 1:
            return mp.inf
        return (_fun.xlogy(a - 1, x) - _fun.xlog1py(a + b, x)
                - _fun.logbeta(a, b))


def cdf(x, a, b):
    """
    Cumulative distribution function of the beta prime distribution.
    """
    with mp.extradps(5):
        a, b = _validate_a_b(a, b)
        x = mp.mpf(x)
        if x < 0:
            return mp.zero
        if x == mp.inf:
            return mp.one
        if x > 1:
            c = mp.betainc(b, a, x1=1/(1+x), x2=1, regularized=True)
        else:
            c = mp.betainc(a, b, x1=0, x2=x/(1+x), regularized=True)
        return c


def sf(x, a, b):
    """
    Survival function of the beta prime distribution.
    """
    with mp.extradps(5):
        a, b = _validate_a_b(a, b)
        x = mp.mpf(x)
        if x < 0:
            return mp.one
        if x == mp.inf:
            return mp.zero
        if x > 1:
            c = mp.betainc(b, a, x1=0, x2=1/(1+x), regularized=True)
        else:
            c = mp.betainc(a, b, x1=x/(1+x), x2=1, regularized=True)
        return c


def invcdf(p, a, b):
    """
    Inverse of the CDF of the beta prime distribution.
    """
    with mp.extradps(5):
        a, b = _validate_a_b(a, b)
        p = _validate_p(p)
        if p == 0:
            return mp.zero
        if p == 1:
            return mp.inf

        x0, x1 = _find_bracket(lambda x: cdf(x, a, b), p, 0, mp.inf)
        if x0 == x1:
            return x0
        x = mp.findroot(lambda x: cdf(x, a, b) - p, x0=(x0, x1),
                        solver='secant')
        return x


def invsf(p, a, b):
    """
    Inverse of the survival function of the beta prime distribution.
    """
    with mp.extradps(5):
        a, b = _validate_a_b(a, b)
        p = _validate_p(p)
        if p == 0:
            return mp.one
        if p == 1:
            return mp.zero

        x0, x1 = _find_bracket(lambda x: sf(x, a, b), p, 0, mp.inf)
        if x0 == x1:
            return x0
        x = mp.findroot(lambda x: sf(x, a, b) - p, x0=(x0, x1),
                        solver='secant')
        return x


def mean(a, b):
    """
    Mean of the beta prime distribution.
    """
    with mp.extradps(5):
        a, b = _validate_a_b(a, b)
        if b <= 1:
            return mp.inf
        return a / (b - 1)


def mode(a, b):
    """
    Mode of the beta prime distribution.
    """
    with mp.extradps(5):
        a, b = _validate_a_b(a, b)
        if a < 1:
            return mp.zero
        return (a - 1) / (b + 1)


def var(a, b):
    """
    Variance of the beta prime distribution.
    """
    with mp.extradps(5):
        a, b = _validate_a_b(a, b)
        if b <= 2:
            return mp.nan
        return (a * (a + b - 1)) / ((b - 2)*(b - 1)**2)


def skewness(a, b):
    """
    Skewness of the beta prime distribution.
    """
    with mp.extradps(5):
        a, b = _validate_a_b(a, b)
        if b <= 3:
            return mp.nan
        t1 = 2 * (2*a + b - 1) / (b - 3)
        t2 = mp.sqrt((b - 2) / (a*(a + b - 1)))
        return t1 * t2


def noncentral_moment(n, a, b):
    """
    Noncentral moment (i.e. raw moment) of the beta prime distribution.

    ``n`` is the order of the moment.  The moment is not defined if
    ``n >= b``; ``nan`` is returned in that case.
    """
    with mp.extradps(5):
        n = _validate_moment_n(n)
        a, b = _validate_a_b(a, b)
        if n >= b:
            return mp.nan
        return mp.beta(a + n, b - n) / mp.beta(a, b)
