"""
Poisson distribution
--------------------

"""

import itertools
from mpmath import mp
from ._common import _validate_counts
from ..stats import mean as _fmean


__all__ = ['support', 'pmf', 'logpmf', 'cdf', 'sf',
           'mean', 'var', 'skewness', 'kurtosis',
           'mle']


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
    return itertools.count(start=0)


def pmf(k, lam):
    """
    Probability mass function of the Poisson distribution.
    """
    if k < 0:
        return mp.zero
    with mp.extradps(5):
        lam = mp.mpf(lam)
        return mp.power(lam, k) * mp.exp(-lam) / mp.factorial(k)


def logpmf(k, lam):
    """
    Natural log of the probability mass function of the binomial distribution.
    """
    if k < 0:
        return mp.ninf
    with mp.extradps(5):
        lam = mp.mpf(lam)
        return k*mp.log(lam) - lam - mp.loggamma(k + 1)


def cdf(k, lam):
    """
    CDF of the Poisson distribution.
    """
    if k < 0:
        return mp.zero
    with mp.extradps(5):
        lam = mp.mpf(lam)
        return mp.gammainc(k + 1, lam, regularized=True)


def sf(k, lam):
    """
    Survival function of the Poisson distribution.
    """
    if k < 0:
        return mp.one
    with mp.extradps(5):
        lam = mp.mpf(lam)
        return mp.gammainc(k + 1, 0, lam, regularized=True)


def mean(lam):
    """
    Mean of the Poisson distribution.
    """
    return mp.mpf(lam)


def var(lam):
    """
    Variance of the Poisson distribution.
    """
    return mp.mpf(lam)


def skewness(lam):
    """
    Skewness of the Poisson distribution.
    """
    return 1/mp.sqrt(mp.mpf(lam))


def kurtosis(lam):
    """
    Excess kurtosis of the Poisson distribution.
    """
    return 1/mp.mpf(lam)


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
