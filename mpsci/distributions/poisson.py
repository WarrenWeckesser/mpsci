"""
Poisson distribution
--------------------

"""

import itertools
from mpmath import mp
from ._common import _validate_x_bounds, _validate_counts
from ..stats import mean as _fmean


__all__ = ['support', 'pmf', 'logpmf', 'cdf', 'sf',
           'mean', 'var', 'skewness', 'kurtosis',
           'nll', 'mle']


def _validate_lam(lam):
    if lam <= 0:
        raise ValueError('lam must be greater than 0')
    return mp.mpf(lam)


def support(lam):
    """
    Support of the Poisson distribution.

    The support is the integers 0, 1, 2, 3, ..., so the support is returned
    as an instance of `itertools.count(start=0)`.

    Examples
    --------
    >>> from mpsci.distributions import poisson
    >>> sup = poisson.support()
    >>> next(sup)
    0
    >>> next(sup)
    1

    """
    lam = _validate_lam(lam)
    return itertools.count(start=0)


@mp.extradps(5)
def pmf(k, lam):
    """
    Probability mass function of the Poisson distribution.
    """
    lam = _validate_lam(lam)
    if k < 0:
        return mp.zero
    return mp.power(lam, k) * mp.exp(-lam) / mp.factorial(k)


@mp.extradps(5)
def logpmf(k, lam):
    """
    Natural log of the probability mass function of the binomial distribution.
    """
    lam = _validate_lam(lam)
    if k < 0:
        return mp.ninf
    return k*mp.log(lam) - lam - mp.loggamma(k + 1)


@mp.extradps(5)
def cdf(k, lam):
    """
    CDF of the Poisson distribution.
    """
    lam = _validate_lam(lam)
    if k < 0:
        return mp.ninf
    return mp.gammainc(k + 1, lam, regularized=True)


@mp.extradps(5)
def sf(k, lam):
    """
    Survival function of the Poisson distribution.
    """
    lam = _validate_lam(lam)
    if k < 0:
        return mp.one
    return mp.gammainc(k + 1, 0, lam, regularized=True)


@mp.extradps(5)
def mean(lam):
    """
    Mean of the Poisson distribution.
    """
    lam = _validate_lam(lam)
    return lam


@mp.extradps(5)
def var(lam):
    """
    Variance of the Poisson distribution.
    """
    lam = _validate_lam(lam)
    return lam


@mp.extradps(5)
def skewness(lam):
    """
    Skewness of the Poisson distribution.
    """
    lam = _validate_lam(lam)
    return 1/mp.sqrt(lam)


@mp.extradps(5)
def kurtosis(lam):
    """
    Excess kurtosis of the Poisson distribution.
    """
    lam = _validate_lam(lam)
    return 1/lam


@mp.extradps(5)
def nll(x, lam, *, counts=None):
    """
    Negative log-likelihood of the Poisson distribution.

    `x` must be a sequence of nonnegative integers.
    """
    lam = _validate_lam(lam)
    x = _validate_x_bounds(x, low=0, high=None, strict_low=False)
    if not all([mp.isint(t) and t >= 0 for t in x]):
        raise ValueError('all values in x must be nonnegative integers')
    counts = _validate_counts(x, counts, expand_none=True)
    return -mp.fsum([count*logpmf(t, lam)
                     for t, count in zip(x, counts)])


@mp.extradps(5)
def mle(x, *, counts=None):
    """
    Maximum likelihood estimate for the Poisson distribution.

    x must be a sequence of numbers that are presumed to be a sample
    from a Poisson distribution.

    Returns lambda, the estimated parameter of the Poisson distribution.
    """
    counts = _validate_counts(x, counts, expand_none=False)
    return _fmean(x, weights=counts)
