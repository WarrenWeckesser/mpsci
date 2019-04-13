"""
The hypergeometric distribution.
"""

import mpmath


__all__ = ['pmf', 'cdf', 'sf', 'support']


def pmf(k, ntotal, ngood, nsample):
    """
    Probability mass function of the hypergeometric distribution.
    """
    nbad = ntotal - ngood
    numer = (ntotal + 1) * mpmath.beta(ntotal - nsample + 1, nsample + 1)
    denom = ((ngood + 1) * (nbad + 1) * mpmath.beta(k + 1, ngood - k + 1) *
             mpmath.beta(nsample - k + 1, nbad - nsample + k + 1))
    pmf = numer / denom
    return pmf


def cdf(k, ntotal, ngood, nsample):
    """
    Cumulative distribution function of the hypergeometric distribution.
    """
    return 1 - sf(k, ntotal, ngood, nsample)


def sf(k, ntotal, ngood, nsample):
    """
    Survival function of the hypergeometric distribution.
    """
    h = mpmath.hyp3f2(1, k + 1 - ngood, k + 1 - nsample, k + 2, ntotal + k + 2 - ngood - nsample, 1)
    sf = (mpmath.binomial(nsample, k + 1) * mpmath.binomial(ntotal - nsample, ngood - k - 1) /
          mpmath.binomial(ntotal, ngood) * h)
    return sf


def support(ntotal, ngood, nsample):
    """
    Support of the hypergeometric distribution.

    Returns
    -------
    sup : range
        The range of integers in the support.  (In Python 3, use list(sup) to
        get the list of integers in the support.)
    p : sequence of mpmath floats
        The probability of each integer in the support.

    Examples
    --------
    >>> sup, pvals = hypergeometric.support(20, 14, 5)
    >>> for k, p in zip(sup, pvals):
    ...     print("{:2} {:10.7f}".format(k, float(p)))
    ... 
     0  0.0003870
     1  0.0135449
     2  0.1173891
     3  0.3521672
     4  0.3873839
     5  0.1291280

    """
    nbad = ntotal - ngood
    p = []
    support = range(max(0, nsample - nbad), min(nsample, ngood) + 1)
    for k in support:
        p.append(pmf(k, ntotal, ngood, nsample))
    return support, p
