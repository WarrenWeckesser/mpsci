"""
Negative hypergeometric distribution
------------------------------------
"""

import mpmath
from ..fun import logbinomial
from .hypergeometric import cdf as hg_cdf, sf as hg_sf


__all__ = ['pmf', 'logpmf', 'cdf', 'sf', 'mean', 'var', 'support']


def pmf(k, ntotal, ngood, untilnbad):
    """
    Probability mass function of the negative hypergeometric distribution.
    """
    with mpmath.extradps(5):
        b1 = mpmath.binomial(k + untilnbad - 1, k)
        b2 = mpmath.binomial(ntotal - untilnbad - k, ngood - k)
        b3 = mpmath.binomial(ntotal, ngood)
        return b1 * (b2 / b3)


def logpmf(k, ntotal, ngood, untilnbad):
    """
    Logarithm of the prob. mass function of the negative hypergeometric distr.
    """
    with mpmath.extradps(5):
        t1 = logbinomial(k + untilnbad - 1, k)
        t2 = logbinomial(ntotal - untilnbad - k, ngood - k)
        t3 = logbinomial(ntotal, ngood)
        return mpmath.fsum([t1, t2, -t3])


def cdf(k, ntotal, ngood, untilnbad):
    """
    Cumulative distribution function of the negative hypergeometric distr.
    """
    if k < 0:
        return mpmath.mp.zero
    if k >= ngood:
        return mpmath.mp.one
    return hg_sf(untilnbad - 1, ntotal, ntotal - ngood, k + 1)


def sf(k, ntotal, ngood, untilnbad):
    """
    Survival function of the negative hypergeometric distribution.
    """
    if k < 0:
        return mpmath.mp.one
    if k >= ngood:
        return mpmath.mp.zero
    return hg_cdf(untilnbad - 1, ntotal, ntotal - ngood, k + 1)


def mean(ntotal, ngood, untilnbad):
    """
    Mean of the negative hypergeometric distribution.
    """
    return mpmath.mpf(untilnbad) * ngood / (ntotal - ngood + 1)


def var(ntotal, ngood, untilnbad):
    """
    Variance of the negative hypergeometric distribution.
    """
    nbad = ntotal - ngood
    r = mpmath.mpf(untilnbad)
    v = (r * (ntotal + 1) * ngood * (mpmath.mp.one - r / (nbad + 1))
         / (nbad + 1) / (nbad + 2))
    return v


def support(ntotal, ngood, untilnbad):
    """
    Support of the negative hypergeometric distribution.

    Returns
    -------
    sup : range
        The range of integers in the support.  (In Python 3, use list(sup) to
        get the list of integers in the support.)
    p : sequence of mpmath floats
        The probability of each integer in the support.

    """
    p = []
    support = range(ngood + 1)
    for k in support:
        p.append(pmf(k, ntotal, ngood, untilnbad))
    return support, p
