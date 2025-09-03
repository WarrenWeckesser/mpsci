"""
Truncated discrete exponential distribution
-------------------------------------------

The distribution `trunc_discrete_exp` is equivalent to SciPy's `boltzmann`
distribution.

The parameters are `lam` > 0 (real parameter) and `n` > 0 (integer parameter).

The support is {0, 1, 2, ..., n - 1}.

The probability mass function is p(k) = C*exp(-lam*k), where C is a normalization
constant.

The distribution is uniform if `lam` == 0.
"""

from mpmath import mp
from ..fun import logbinomial, logbeta
from ._common import _validate_x_bounds, Initial, _validate_counts, _validate_k


__all__ = ['support', 'pmf', 'logpmf', 'cdf', 'sf',
           'mean', 'var', 'skewness', 'kurtosis',
           'nll', 'mle']


def _validate_params(lam, n):
    if n != int(n):
        raise ValueError('n must be an integer')
    if n < 0:
        raise ValueError('n must be nonngative')
    if lam < 0:
        raise ValueError('lam must be nonnegative')
    return mp.mpf(lam), int(n)


def support(lam, n):
    """
    Support of the truncated discrete exponential distribution.

    The support is the integers 0, 1, 2, ..., n - 1; this is implemented
    by returning `range(n)`.  That is, the return value is the `range`
    instance, not a sequence.

    Examples
    --------
    >>> from mpsci.distributions import trunc_discrete_exp
    >>> sup = trunc_discrete_exp.support(2.5, 6)
    >>> [k for k in sup]
    [0, 1, 2, 3, 4, 5]

    """
    lam, n = _validate_params(lam, n)
    return range(n)


def logpmf(k, lam, n):
    """
    Logarithm of PMF of the truncated discrete exponential distribution.
    """
    k = _validate_k(k)
    with mp.extradps(5):
        lam, n = _validate_params(lam, n)
        if k < 0:
            return mp.ninf
        if k >= n:
            return mp.ninf
        if lam == 0:
            return -mp.log(n)
        return mp.log1p(-mp.exp(-lam)) - lam*k - mp.log1p(-mp.exp(-lam*n))


def pmf(k, lam, n):
    """
    Probability mass function of the truncated discrete exponential distribution.
    """
    k = _validate_k(k)
    with mp.extradps(5):
        lam, n = _validate_params(lam, n)
        if k < 0:
            return mp.zero
        if k >= n:
            return mp.zero
        if lam == 0:
            return mp.one/n
        return mp.exp(logpmf(k, lam, n))


def cdf(k, lam, n):
    """
    Cumulative distribution function of the truncated discrete exponential distribution.
    """
    k = _validate_k(k)
    with mp.extradps(5):
        lam, n = _validate_params(lam, n)
        if k < 0:
            return mp.zero
        if k >= n:
            return mp.one
        if lam == 0:
            return (k + mp.one)/n
        return mp.expm1(-lam*(k + 1)) / mp.expm1(-lam*n)


def sf(k, lam, n):
    """
    Survival function of the truncated discrete exponential distribution.
    """
    k = _validate_k(k)
    with mp.extradps(5):
        lam, n = _validate_params(lam, n)
        if k < 0:
            return mp.one
        if k >= n:
            return mp.zero
        if lam == 0:
            return (n - k - mp.one)/n
        return mp.expm1(lam*(n - k - 1)) / mp.expm1(lam*n)


def mean(lam, n):
    """
    Mean of the truncated discrete exponential distribution.
    """
    with mp.extradps(5):
        lam, n = _validate_params(lam, n)
        if lam == 0:
            return (n - mp.one)/2
        nlam = n*lam
        return -mp.exp(-lam)/mp.expm1(-lam) + n*mp.exp(-nlam)/mp.expm1(-nlam)


def var(lam, n):
    """
    Variance of the truncated discrete exponential distribution.
    """
    with mp.extradps(5):
        lam, n = _validate_params(lam, n)
        if lam == 0:
            return (mp.mpf(n)**2 - 1)/12
        nlam = n*lam
        return mp.exp(-lam)/mp.expm1(-lam)**2 - n**2 * mp.exp(-nlam)/mp.expm1(-nlam)**2
