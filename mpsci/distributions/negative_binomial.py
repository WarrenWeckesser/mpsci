"""
Negative binomial distribution
------------------------------

There are several different ways to parameterize the negative binomial
distribution.  Here, the quantiles are the number of "successes" that
occur when draws from a binomial distribution are made repeatedly until
the number of "failures" drawn is `r`.  `p` is the probability of drawing
a "success".
"""

import mpmath
from ..fun import logbinomial, xlogy, xlog1py


__all__ = ['pmf', 'logpmf', 'sf', 'cdf', 'mean', 'var']


def logpmf(k, r, p):
    """
    Log of the probability mass function of the negative binomial distribution.

    Parameters
    ----------
    r : int
        Number of failures until the experiment is stopped.
    p : float
        Probability of success.
    """
    with mpmath.extradps(5):
        k = mpmath.mpf(k)
        r = mpmath.mpf(r)
        p = mpmath.mpf(p)
        return logbinomial(k + r - 1, k) + xlog1py(r, -p) + xlogy(k, p)


def pmf(k, r, p):
    """
    Probability mass function of the negative binomial distribution.

    Parameters
    ----------
    r : int
        Number of failures until the experiment is stopped.
    p : float
        Probability of success.
    """
    return mpmath.exp(logpmf(k, r, p))


def sf(k, r, p):
    """
    Survival function of the negative binomial distribution.

    Parameters
    ----------
    r : int
        Number of failures until the experiment is stopped.
    p : float
        Probability of success.
    """
    with mpmath.extradps(5):
        k = mpmath.mpf(k)
        r = mpmath.mpf(r)
        p = mpmath.mpf(p)
        return mpmath.betainc(k + 1, r, 0, p, regularized=True)


def cdf(k, r, p):
    """
    Cumulative distribution function of the negative binomial distribution.

    Parameters
    ----------
    r : int
        Number of failures until the experiment is stopped.
    p : float
        Probability of success.
    """
    with mpmath.extradps(5):
        k = mpmath.mpf(k)
        r = mpmath.mpf(r)
        p = mpmath.mpf(p)
        return mpmath.betainc(k + 1, r, p, 1, regularized=True)


def mean(r, p):
    """
    Mean of the negative binomial distribution.

    Parameters
    ----------
    r : int
        Number of failures until the experiment is stopped.
    p : float
        Probability of success.
    """
    with mpmath.extradps(5):
        r = mpmath.mpf(r)
        p = mpmath.mpf(p)
        return p*r / (1 - p)


def var(r, p):
    """
    Variance of the negative binomial distribution.

    Parameters
    ----------
    r : int
        Number of failures until the experiment is stopped.
    p : float
        Probability of success.
    """
    with mpmath.extradps(5):
        r = mpmath.mpf(r)
        p = mpmath.mpf(p)
        return p*r / (1 - p)**2
