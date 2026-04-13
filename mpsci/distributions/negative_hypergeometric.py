"""
Negative hypergeometric distribution
------------------------------------

The parameterization used here is the same as that of the Wikipedia article
https://en.wikipedia.org/wiki/Negative_hypergeometric_distribution: `ntotal = N`,
`ngood = K`, and `untilnbad = r`.

"""

from mpmath import mp
from ..fun import logbinomial
from .hypergeometric import cdf as hg_cdf, sf as hg_sf


__all__ = ['pmf', 'logpmf', 'cdf', 'sf', 'mean', 'mode', 'var',
           'support', 'support_pmf']


def _validate(ntotal, ngood, untilnbad):
    if ntotal < 0 or ngood < 0 or untilnbad < 0:
        raise ValueError('all distribution parameters must be nonnegative')
    if ngood > ntotal:
        raise ValueError('ngood must not be greater than ntotal')
    if ntotal - ngood < untilnbad:
        raise ValueError('untilnbad must not be greater than ntotal - ngood')


@mp.extradps(5)
def pmf(k, ntotal, ngood, untilnbad):
    """
    Probability mass function of the negative hypergeometric distribution.
    """
    _validate(ntotal, ngood, untilnbad)

    if k < 0 or k > ngood:
        return mp.zero

    b1 = mp.binomial(k + untilnbad - 1, k)
    b2 = mp.binomial(ntotal - untilnbad - k, ngood - k)
    b3 = mp.binomial(ntotal, ngood)
    return b1 * (b2 / b3)


@mp.extradps(5)
def logpmf(k, ntotal, ngood, untilnbad):
    """
    Logarithm of the prob. mass function of the negative hypergeometric distr.
    """
    _validate(ntotal, ngood, untilnbad)

    if k < 0 or k > ngood:
        return mp.ninf

    t1 = logbinomial(k + untilnbad - 1, k)
    t2 = logbinomial(ntotal - untilnbad - k, ngood - k)
    t3 = logbinomial(ntotal, ngood)
    return mp.fsum([t1, t2, -t3])


def cdf(k, ntotal, ngood, untilnbad):
    """
    Cumulative distribution function of the negative hypergeometric distr.
    """
    _validate(ntotal, ngood, untilnbad)

    if k < 0:
        return mp.zero
    if k >= ngood:
        return mp.one
    return hg_sf(untilnbad - 1, ntotal, ntotal - ngood, k + untilnbad)


def sf(k, ntotal, ngood, untilnbad):
    """
    Survival function of the negative hypergeometric distribution.
    """
    _validate(ntotal, ngood, untilnbad)

    if k < 0:
        return mp.one
    if k >= ngood:
        return mp.zero
    return hg_cdf(untilnbad - 1, ntotal, ntotal - ngood, k + untilnbad)


@mp.extradps(5)
def mean(ntotal, ngood, untilnbad):
    """
    Mean of the negative hypergeometric distribution.
    """
    _validate(ntotal, ngood, untilnbad)

    return mp.mpf(untilnbad) * ngood / (ntotal - ngood + 1)


@mp.extradps(5)
def mode(ntotal, ngood, untilnbad):
    """
    Mode of the negative hypergeometric distribution.
    """
    _validate(ntotal, ngood, untilnbad)

    if untilnbad == 0 or ngood == 0:
        # In this case, the support is just the one point [0].
        return 0

    ngood = mp.mpf(ngood)
    untilnbad = mp.mpf(untilnbad)

    if ntotal > untilnbad * (ngood + 1):
        # The PMF is decreasing.
        return 0
    t = ngood + untilnbad - 1
    if t > (ntotal - t) * ngood:
        # The PMF is increasing.
        return ngood
    return int(mp.ceil((untilnbad * (ngood + 1) - ntotal) / (ntotal - ngood - 1)))


@mp.extradps(5)
def var(ntotal, ngood, untilnbad):
    """
    Variance of the negative hypergeometric distribution.
    """
    _validate(ntotal, ngood, untilnbad)

    nbad = ntotal - ngood
    r = mp.mpf(untilnbad)
    v = (r * (ntotal + 1) * ngood * (mp.one - r / (nbad + 1))
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
    """
    _validate(ntotal, ngood, untilnbad)

    if untilnbad == 0:
        m = 1
    else:
        m = ngood + 1
    return range(m)


def support_pmf(ntotal, ngood, untilnbad):
    """
    Support and PMF of the negative hypergeometric distribution.

    Returns
    -------
    sup : range
        The range of integers in the support.  (In Python 3, use list(sup) to
        get the list of integers in the support.)
    p : sequence of mpmath floats
        The probability of each integer in the support.

    """
    _validate(ntotal, ngood, untilnbad)

    if untilnbad == 0:
        m = 1
    else:
        m = ngood + 1
    p = []
    sup = range(m)
    for k in sup:
        p.append(pmf(k, ntotal, ngood, untilnbad))
    return sup, p
