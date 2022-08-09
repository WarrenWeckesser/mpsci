"""
Poisson distribution
--------------------

"""

import mpmath


__all__ = ['pmf', 'logpmf', 'cdf', 'sf', 'mean', 'var', 'mle']


def pmf(k, lam):
    """
    Probability mass function of the Poisson distribution.
    """
    if k < 0:
        return mpmath.mp.zero
    with mpmath.extradps(5):
        lam = mpmath.mpf(lam)
        return mpmath.power(lam, k) * mpmath.exp(-lam) / mpmath.factorial(k)


def logpmf(k, lam):
    """
    Natural log of the probability mass function of the binomial distribution.
    """
    if k < 0:
        return -mpmath.mp.inf
    with mpmath.extradps(5):
        lam = mpmath.mpf(lam)
        return k*mpmath.log(lam) - lam - mpmath.loggamma(k + 1)


def cdf(k, lam):
    """
    CDF of the Poisson distribution.
    """
    if k < 0:
        return mpmath.mp.zero
    with mpmath.extradps(5):
        lam = mpmath.mpf(lam)
        return mpmath.gammainc(k + 1, lam, regularized=True)


def sf(k, lam):
    """
    Survival function of the Poisson distribution.
    """
    if k < 0:
        return mpmath.mp.one
    with mpmath.extradps(5):
        lam = mpmath.mpf(lam)
        return mpmath.gammainc(k + 1, 0, lam, regularized=True)


def mean(lam):
    """
    Mean of the Poisson distribution.
    """
    return mpmath.mpf(lam)


def var(lam):
    """
    Variance of the Poisson distribution.
    """
    return mpmath.mpf(lam)


def mle(x):
    """
    Maximum likelihood estimate for the Poisson distribution.

    x must be a sequence of numbers that are presumed to be a sample
    from a Poisson distribution.

    Returns lambda, the estimated parameter of the Poisson distribution.
    """
    with mpmath.extradps(5):
        return mpmath.fsum(x) / len(x)
