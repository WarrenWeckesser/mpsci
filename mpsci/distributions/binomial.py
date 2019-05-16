"""
Binomial distribution
---------------------

"""

import mpmath
from ..fun import logbinomial


__all__ = ['pmf', 'logpmf', 'cdf', 'mean', 'var']


def _validate_np(n, p):
    if p < 0 or p > 1:
        raise ValueError('p must be in the range [0, 1]')
    if n < 0:
        raise ValueError('n must be a nonnegative integer.')
    return


def pmf(k, n, p):
    """
    Probability mass function of the binomial distribution.
    """
    _validate_np(n, p)
    with mpmath.extradps(5):
        p = mpmath.mpf(p)
        return (mpmath.binomial(n, k) *
                mpmath.power(p, k) *
                mpmath.power(1 - p, n - k))


def logpmf(k, n, p):
    """
    Natural log of the probability mass function of the binomial distribution.
    """
    _validate_np(n, p)
    with mpmath.extradps(5):
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
    _validate_np(n, p)
    if method not in ['sumpmf', 'incbeta']:
        raise ValueError('method must be "sum" or "incbeta"')
    if method == 'incbeta':
        with mpmath.extradps(5):
            p = mpmath.mpf(p)
            # XXX For large values of k and/or n, betainc fails. The failure
            # occurs in one of the hypergeometric functions.
            return mpmath.betainc(n - k, k + 1, x1=0, x2=1 - p,
                                  regularized=True)
    else:
        # method is "sum"
        with mpmath.extradps(5):
            c = mpmath.fsum([mpmath.exp(logpmf(t, n, p))
                             for t in range(k + 1)])
            return c


def mean(n, p):
    """
    Mean of the binomial distribution.
    """
    _validate_np(n, p)
    with mpmath.extradps(5):
        n = mpmath.mpf(n)
        p = mpmath.mpf(p)
        return n*p


def var(n, p):
    """
    Variance of the binomial distribution.
    """
    _validate_np(n, p)
    with mpmath.extradps(5):
        n = mpmath.mpf(n)
        p = mpmath.mpf(p)
        return n * p * (1 - p)
