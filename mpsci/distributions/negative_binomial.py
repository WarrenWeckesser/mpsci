"""
Negative binomial distribution
------------------------------

There are several different ways to parameterize the negative binomial
distribution.  Here, the quantiles are the number of "successes" that
occur when draws from a binomial distribution are made repeatedly until
the number of "failures" drawn is `r`.  `p` is the probability of drawing
a "success".
"""

from mpmath import mp
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
    with mp.extradps(5):
        k = mp.mpf(k)
        r = mp.mpf(r)
        p = mp.mpf(p)
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
    return mp.exp(logpmf(k, r, p))


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
    with mp.extradps(5):
        k = mp.mpf(k)
        r = mp.mpf(r)
        p = mp.mpf(p)
        return mp.betainc(k + 1, r, 0, p, regularized=True)


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
    with mp.extradps(5):
        k = mp.mpf(k)
        r = mp.mpf(r)
        p = mp.mpf(p)
        return mp.betainc(k + 1, r, p, 1, regularized=True)


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
    with mp.extradps(5):
        r = mp.mpf(r)
        p = mp.mpf(p)
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
    with mp.extradps(5):
        r = mp.mpf(r)
        p = mp.mpf(p)
        return p*r / (1 - p)**2
