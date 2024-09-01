"""
Log-series distribution
-----------------------

This discrete distributions is also known as the logarithmic
distribution [1]_.

.. [1] Logarithmic distribution,
       https://en.wikipedia.org/wiki/Logarithmic_distribution
"""

import itertools
from mpmath import mp
from mpsci.stats import mean as _fmean
from ._common import _validate_p, _validate_counts


__all__ = ['support', 'pmf', 'logpmf', 'cdf', 'sf', 'mean', 'var', 'mode',
           'skewness', 'kurtosis', 'nll', 'mle']


def support(p):
    """
    Support of the log-series distribution.

    The support is the integers 1, 2, 3, ..., so the support is returned
    as an instance of `itertools.count(start=1)`.

    Examples
    --------
    >>> from mpsci.distributions import logseries
    >>> sup = logseries.support()
    >>> next(sup)
    1
    >>> next(sup)
    2

    """
    return itertools.count(start=1)


def pmf(k, p):
    """
    Probability mass function of the log-series distribution.
    """
    with mp.extradps(5):
        p = _validate_p(p)
        if k < 1:
            return mp.zero
        return mp.exp(logpmf(k, p))


def logpmf(k, p):
    """
    Natural log of the PMF of the log-series distribution.
    """
    with mp.extradps(5):
        p = _validate_p(p)
        if k < 1:
            return mp.ninf
        return k*mp.log(p) - mp.log(k) - mp.log(-mp.log1p(-p))


def cdf(k, p):
    """
    CDF of the log-series distribution.
    """
    with mp.extradps(5):
        p = _validate_p(p)
        if k < 1:
            return mp.zero
        return 1 + mp.betainc(k + 1, 0, 0, p) / mp.log1p(-p)


def sf(k, p):
    """
    Survival function of the log-series distribution.
    """
    with mp.extradps(5):
        p = _validate_p(p)
        if k < 1:
            return mp.one
        return -mp.betainc(k + 1, 0, 0, p) / mp.log1p(-p)


def mean(p):
    """
    Mean of the log-series distribution.
    """
    with mp.extradps(5):
        p = _validate_p(p)
        return p / (p - 1) / mp.log1p(-p)


def var(p):
    """
    Variance of the log-series distribution.
    """
    with mp.extradps(5):
        p = _validate_p(p)
        l1p = mp.log1p(-p)
        return -(p*(p + l1p)) / (1 - p)**2 / l1p**2


def mode(p):
    """
    Mode of the log-series distribution.
    """
    p = _validate_p(p)
    return mp.one


def skewness(p):
    """
    Skewness of the log-series distribution.
    """
    with mp.extradps(5):
        p = _validate_p(p)
        r = mp.log1p(-p)
        s = p + r
        num = p*(2*p + 3*r) + (1 + p)*r**2
        den = -mp.sqrt(-p*s)*s
        return num/den


def kurtosis(p):
    """
    Excess kurtosis of the log-series distribution.
    """
    with mp.extradps(5):
        p = _validate_p(p)
        r = mp.log1p(-p)
        r2 = r*r
        r3 = r2*r
        num = p*(p*(-6*p + r*(r*(-r - 4) - 12)) + r2*(-4*r - 7)) - r3
        den = p*(p + r)**2
        return num/den


@mp.extradps(5)
def nll(x, p, *, counts=None):
    """
    Negative log-likelihood of the log-series distribution.
    """
    p = _validate_p(p)
    if not all([mp.isint(t) and t >= 1 for t in x]):
        raise ValueError('all values in x must be integers greater than 0')
    counts = _validate_counts(x, counts, expand_none=True)
    return -mp.fsum([count*logpmf(t, p)
                     for t, count in zip(x, counts)])


def _approx_inv_mle_func(m):
    # For m sufficiently large (i.e. p close to 1).
    pa = 1 + 1/(m*mp.lambertw(-1/m, k=-1).real)
    m = m/pa
    # This correction usually improves the approximation.
    pa2 = 1 + 1/(m*mp.lambertw(-1/m, k=-1).real)
    if pa2 < 1:
        return pa2
    return pa


def _mle_func(p):
    return -p/((1 - p)*mp.log1p(-p))


@mp.extradps(5)
def mle(x, *, counts=None):
    """
    Maximum likelihood estimation for the log-series distribution.

    Examples
    --------
    >>> from mpsci.distributions import logseries
    >>> from mpmath import mp
    >>> mp.dps = 40

    >>> x = [1, 1, 1, 1, 3, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2]
    >>> logseries.mle(x)
    mpf('0.3500385397570795760273865807843594135596862')

    >>> values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    >>> counts = [10, 4, 5, 2, 0, 2, 5, 1, 4, 0, 3]
    >>> logseries.mle(values, counts=counts)
    mpf('0.9207611550739465028229025367922360801454828')

    """
    with mp.extradps(mp.dps):
        m = _fmean(x, weights=counts)
        if m == 1:
            return mp.zero
        if m > 2.65:
            pa = _approx_inv_mle_func(m)
            p = mp.findroot(lambda t: _mle_func(t) - m, pa,
                            method='newton').real
            return p
        else:
            slope = 0.819/1.65
            low = slope*(m - 1)
            high = min(2*(m - 1), 0.8196)
            p = mp.findroot(lambda t: _mle_func(t) - m, [low, high],
                            method='ridder')
            return p
