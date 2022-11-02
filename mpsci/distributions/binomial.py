"""
Binomial distribution
---------------------

"""

import mpmath
from ._common import _validate_p
from ..fun import logbinomial


__all__ = ['pmf', 'logpmf', 'cdf', 'sf', 'mean', 'var']


def _validate_np(n, p):
    p = _validate_p(p)
    if n < 0:
        raise ValueError('n must be a nonnegative integer.')
    return n, p


def pmf(k, n, p):
    """
    Probability mass function of the binomial distribution.
    """
    with mpmath.extradps(5):
        n, p = _validate_np(n, p)
        return (mpmath.binomial(n, k) *
                mpmath.power(p, k) *
                mpmath.power(1 - p, n - k))


def logpmf(k, n, p):
    """
    Natural log of the probability mass function of the binomial distribution.
    """
    with mpmath.extradps(5):
        n, p = _validate_np(n, p)
        return (logbinomial(n, k)
                + k*mpmath.log(p)
                + mpmath.fsum([n, -k])*mpmath.log1p(-p))


def cdf(k, n, p, method='incbeta'):
    """
    Cumulative distribution function of the binomial distribution.

    `method` must be either "sumpmf" or "incbeta".  When `method` is "sumpmf",
    the CDF is computed with a simple sum of the PMF values.  When `method`
    is "incbeta", the incomplete beta function is used. This method is
    generally faster than the "sumpmf" method, but for large values of k
    or n, the incomplete beta function of mpmath might fail.
    """
    if method not in ['sumpmf', 'incbeta']:
        raise ValueError('method must be "sum" or "incbeta"')
    if method == 'incbeta':
        with mpmath.extradps(5):
            n, p = _validate_np(n, p)
            # XXX For large values of k and/or n, betainc fails. The failure
            # occurs in one of the hypergeometric functions.
            return mpmath.betainc(n - k, k + 1, x1=0, x2=1 - p,
                                  regularized=True)
    else:
        # method is "sumpmf"
        with mpmath.extradps(5):
            n, p = _validate_np(n, p)
            c = mpmath.fsum([mpmath.exp(logpmf(t, n, p))
                             for t in range(k + 1)])
            return c


def sf(k, n, p, method='incbeta'):
    """
    Survival function of the binomial distribution.

    `method` must be either "sumpmf" or "incbeta".  When `method` is "sumpmf",
    the survival function is computed with a simple sum of the PMF values.
    When `method` is "incbeta", the incomplete beta function is used. This
    method is generally faster than the "sumpmf" method, but for large values
    of k or n, the incomplete beta function of mpmath might fail.
    """
    if method not in ['sumpmf', 'incbeta']:
        raise ValueError('method must be "sum" or "incbeta"')
    if method == 'incbeta':
        with mpmath.extradps(5):
            n, p = _validate_np(n, p)
            # XXX For large values of k and/or n, betainc fails. The failure
            # occurs in one of the hypergeometric functions.
            return mpmath.betainc(n - k, k + 1, x1=1-p, x2=1,
                                  regularized=True)
    else:
        # method is "sumpmf"
        with mpmath.extradps(5):
            n, p = _validate_np(n, p)
            c = mpmath.fsum([mpmath.exp(logpmf(t, n, p))
                             for t in range(k + 1, n + 1)])
            return c


def mean(n, p):
    """
    Mean of the binomial distribution.
    """
    with mpmath.extradps(5):
        n, p = _validate_np(n, p)
        return n*p


def var(n, p):
    """
    Variance of the binomial distribution.
    """
    with mpmath.extradps(5):
        n, p = _validate_np(n, p)
        return n * p * (1 - p)
