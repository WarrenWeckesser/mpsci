"""
Hypergeometric distribution
---------------------------
"""

from mpmath import mp
from ..fun import logbeta


__all__ = ['pmf', 'logpmf', 'cdf', 'sf', 'support']


def _validate(ntotal, ngood, nsample):
    if ntotal < 0 or ngood < 0 or nsample < 0:
        raise ValueError(f'ntotal ({ntotal}), ngood ({ngood}) and nsample '
                         f'({nsample}) must be nonnegative.')
    if ngood > ntotal:
        raise ValueError(f'ngood ({ngood}) must not be greater than '
                         f'ntotal ({ntotal})')
    if nsample > ntotal:
        raise ValueError(f'nsample ({nsample}) must not be greater than '
                         f'ntotal ({ntotal})')


def pmf(k, ntotal, ngood, nsample):
    """
    Probability mass function of the hypergeometric distribution.
    """
    _validate(ntotal, ngood, nsample)
    nbad = ntotal - ngood
    numer = (ntotal + 1) * mp.beta(ntotal - nsample + 1, nsample + 1)
    denom = ((ngood + 1) * (nbad + 1) * mp.beta(k + 1, ngood - k + 1) *
             mp.beta(nsample - k + 1, nbad - nsample + k + 1))
    pmf = numer / denom
    return pmf


def logpmf(k, ntotal, ngood, nsample):
    """
    Logarithm of the PMF of the hypergeometric distribution.

    `logpmf` computes the natural logarithm of the probability mass function
    of the hypergeometric distribution.
    """
    _validate(ntotal, ngood, nsample)
    nbad = ntotal - ngood
    with mp.extradps(5):
        # numerator terms
        terms = [mp.log(ntotal + 1),
                 logbeta(ntotal - nsample + 1, nsample + 1)]
        # denominator terms
        terms.extend([-mp.log(ngood + 1),
                      -mp.log(nbad + 1),
                      -logbeta(k + 1, ngood - k + 1),
                      -logbeta(nsample - k + 1, nbad - nsample + k + 1)])
        return mp.fsum(terms)


def cdf(k, ntotal, ngood, nsample):
    """
    Cumulative distribution function of the hypergeometric distribution.
    """
    _validate(ntotal, ngood, nsample)
    return 1 - sf(k, ntotal, ngood, nsample)


def sf(k, ntotal, ngood, nsample):
    """
    Survival function of the hypergeometric distribution.
    """
    _validate(ntotal, ngood, nsample)
    h = mp.hyp3f2(1, k + 1 - ngood, k + 1 - nsample, k + 2,
                  ntotal + k + 2 - ngood - nsample, 1)
    num = (mp.binomial(nsample, k + 1) *
           mp.binomial(ntotal - nsample, ngood - k - 1))
    den = mp.binomial(ntotal, ngood)
    sf = (num / den) * h
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
    _validate(ntotal, ngood, nsample)
    nbad = ntotal - ngood
    p = []
    support = range(max(0, nsample - nbad), min(nsample, ngood) + 1)
    for k in support:
        p.append(pmf(k, ntotal, ngood, nsample))
    return support, p
