"""
Beta probability distribution
-----------------------------

"""
import mpmath
from .. import fun as _fun
from ..stats import mean as _mean, var as _var


__all__ = ['pdf', 'logpdf', 'cdf', 'invcdf', 'sf', 'invsf', 'interval_prob',
           'mean', 'mle', 'mom']


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
    return _fun.xlogy(a - 1, x) + _fun.xlog1py(b - 1, -x) - _fun.logbeta(a, b)


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


def interval_prob(x1, x2, a, b):
    """
    Compute the probability of x in [x1, x2] for the beta distribution.

    Mathematically, this is the same as

        beta.cdf(x2, a, b) - beta.cdf(x1, a, b)

    but when the two CDF values are nearly equal, this function will give
    a more accurate result.

    x1 must be less than or equal to x2.
    """
    if x1 > x2:
        raise ValueError('x1 must not be greater than x2')

    with mpmath.extradps(5):
        x1 = mpmath.mpf(x1)
        x2 = mpmath.mpf(x2)
        a = mpmath.mpf(a)
        b = mpmath.mpf(b)
        return mpmath.betainc(a, b, x1, x2, regularized=True)


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


def _beta_mle_func(a, b, n, s):
    psiab = mpmath.digamma(a + b)
    return s - n * (-psiab + mpmath.digamma(a))


def _validate_x(x):
    if any((t <= 0 or t >= 1) for t in x):
        raise ValueError('Values in x must greater than 0 and less than 1')


def mle(x, a=None, b=None):
    """
    Maximum likelihood estimation for the beta distribution.
    """
    _validate_x(x)
    n = len(x)
    xbar = _mean(x)

    if a is None and b is None:
        # Fit both parameters
        s1 = mpmath.fsum(mpmath.log(t) for t in x)
        s2 = mpmath.fsum(mpmath.log1p(-t) for t in x)
        fac = xbar * (mpmath.mp.one - xbar) / _var(x) - 1
        a0 = xbar * fac
        b0 = (mpmath.mp.one - xbar) * fac
        a1, b1 = mpmath.findroot([lambda a, b: _beta_mle_func(a, b, n, s1),
                                  lambda a, b: _beta_mle_func(b, a, n, s2)],
                                 [a0, b0])
        return a1, b1

    if a is not None and b is not None:
        return a, b

    swap = False
    if b is None:
        swap = True
        b = a
        x = [mpmath.mp.one - t for t in x]
        xbar = mpmath.mp.one - xbar

    # Fit the a parameter, with b given.
    s1 = mpmath.fsum(mpmath.log(t) for t in x)
    p0 = b * xbar / (mpmath.mp.one - xbar)
    p1 = mpmath.findroot(lambda a: _beta_mle_func(a, b, n, s1), p0)
    if swap:
        return a, p1
    else:
        return p1, b


def mom(x):
    """
    Method of moments parameter estimation for the beta distribution.

    x must be a sequence of numbers from the interval (0, 1).

    Returns (a, b).
    """
    _validate_x(x)
    with mpmath.extradps(5):
        M1 = _mean(x)
        M2 = _mean([mpmath.mpf(t)**2 for t in x])
        c = (M1 - M2) / (M2 - M1**2)
        a = M1*c
        b = (1 - M1)*c
        return a, b
