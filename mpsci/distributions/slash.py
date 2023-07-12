"""
Slash distribution
------------------

See https://en.wikipedia.org/wiki/Slash_distribution for details.

According to the wikipedia article, the distribution was named in [1]_.

.. [1] Rogers, W. H.; Tukey, J. W. (1972). "Understanding some long-tailed
       symmetrical distributions". Statistica Neerlandica. 26 (3): 211–226.
       doi:10.1111/j.1467-9574.1972.tb00191.x.

"""

from mpmath import mp
from ._common import _validate_p


__all__ = ['pdf', 'logpdf', 'cdf', 'sf', 'invcdf', 'invsf']


# This is a fuzzy threshold, so using a Python float is OK.
_delta_x2_threshold = 9e-6


def _delta(x):
    """
    Compute ϕ(0) - ϕ(x), where ϕ is the PDF of the normal distribution.
    """
    x2 = x*x
    if x2 < _delta_x2_threshold:
        # When x is small, use this:
        #     ϕ(0) - ϕ(x) = (1 - exp(-x**2/2))/sqrt(2*pi)
        #                 = -expm1(-x**2/x)/sqrt(2*pi)
        # (An alternative would be to use a Taylor or Padé
        # approximation.)
        # This formula is not used for all x, because numerical
        # experiments showed that when x2 exceeds the threshold,
        # `npdf(0) - npdf(x)` is more accurate.
        delta = -mp.expm1(-x2/2)/mp.sqrt(2*mp.pi)
    else:
        delta = mp.npdf(0) - mp.npdf(x)
    return delta


def pdf(x):
    """
    Probability density function of the slash distribution.
    """
    with mp.extradps(5):
        if x == 0:
            return 1/(2*mp.sqrt(2*mp.pi))
        x = mp.mpf(x)
        return _delta(x)/(x**2)


def logpdf(x):
    """
    Natural logarithm of the PDF of the slash distribution.
    """
    with mp.extradps(5):
        if x == 0:
            return mp.log(pdf(0))
        x = mp.mpf(x)
        delta = _delta(x)
        return mp.log(delta) - 2*mp.log(mp.absmax(x))


def cdf(x):
    """
    Cumulative distribution function for the slash distribution.
    """
    with mp.extradps(5):
        if x == 0:
            return mp.one/2
        x = mp.mpf(x)
        delta = _delta(x)
        return mp.ncdf(x) - delta/x


def sf(x):
    """
    Survival function for the slash distribution.
    """
    with mp.extradps(5):
        if x == 0:
            return mp.one/2
        x = mp.mpf(x)
        delta = _delta(x)
        return mp.ncdf(-x) + delta/x


def invcdf(p):
    """
    Inverse of the CDF of the slash distribution.

    Also known as the quantile function.

    This function numerically inverts the CDF function so it
    may be slow, and in some cases it may fail to find a solution.
    """
    with mp.extradps(5):
        _npdf0 = mp.npdf(0)
        p = _validate_p(p)
        if p == 0:
            return mp.ninf
        if p == 1:
            return mp.inf
        if p == 0.5:
            return mp.zero
        if p > 0.5:
            x0 = _npdf0/(1 - p)
        else:
            x0 = -_npdf0/p
        return mp.findroot(lambda x: cdf(x) - p, x0=x0)


def invsf(p):
    """
    Inverse of the survival function of the slash distribution.

    This function numerically inverts the survival function so it
    may be slow, and in some cases it may fail to find a solution.
    """
    with mp.extradps(5):
        _npdf0 = mp.npdf(0)
        p = _validate_p(p)
        if p == 0:
            return mp.inf
        if p == 1:
            return mp.ninf
        if p == 0.5:
            return mp.zero
        if p > 0.5:
            x0 = -_npdf0/(1 - p)
        else:
            x0 = _npdf0/p
        return mp.findroot(lambda x: sf(x) - p, x0=x0)
