"""
Inverse gamma distribution
--------------------------

This implementation uses the same parametrization as the SciPy
implementation in `scipy.stats.invgamma`. `loc` and `scale` are the standard
location and scale parameters, but typically discussions and implementations
of the inverse gamma distribution do not include the location parameter that
we include here and in SciPy (see, for example, the wikipedia article [1]_).

.. [1] Inverse-gamma distribution, Wikipedia,
       https://en.wikipedia.org/wiki/Inverse-gamma_distribution

"""

from mpmath import mp
from ._common import _validate_p, _validate_moment_n


__all__ = ['pdf', 'logpdf', 'cdf', 'invcdf', 'sf', 'invsf',
           'mean', 'mode', 'var', 'entropy', 'noncentral_moment']


def _validate_params(a, loc, scale):
    if a <= 0:
        raise ValueError('m must be positive')
    if scale <= 0:
        raise ValueError('scale must be positive')
    a = mp.mpf(a)
    loc = mp.mpf(loc)
    scale = mp.mpf(scale)
    return a, loc, scale


def pdf(x, a, loc=0, scale=1):
    """
    PDF for the inverse gamma distribution.
    """
    with mp.extradps(5):
        a, loc, scale = _validate_params(a, loc, scale)
        x = mp.mpf(x)
        if x <= loc:
            return mp.zero
        return mp.exp(logpdf(x, a, loc, scale))


def logpdf(x, a, loc=0, scale=1):
    """
    Logarithm of the PDF for the inverse gamma distribution.
    """
    with mp.extradps(5):
        a, loc, scale = _validate_params(a, loc, scale)
        x = mp.mpf(x)
        if x <= loc:
            return mp.ninf
        z = (x - loc)/scale
        logp = -mp.log(scale) - mp.loggamma(a) - (a + 1)*mp.log(z) - 1/z
        return logp


def cdf(x, a, loc=0, scale=1):
    """
    CDF for the inverse gamma distribution.
    """
    with mp.extradps(5):
        a, loc, scale = _validate_params(a, loc, scale)
        x = mp.mpf(x)
        if x <= loc:
            return mp.zero
        z = (x - loc)/scale
        return mp.gammainc(a, 1/z, mp.inf, regularized=True)


def invcdf(p, a, loc=0, scale=1):
    """
    Inverse of the CDF for the inverse gamma distribution.
    """
    with mp.extradps(5):
        a, loc, scale = _validate_params(a, loc, scale)
        p = _validate_p(p)
        if p == 0:
            return loc
        if p == 1:
            return mp.inf
        # WIP... not updated yet...
        x0 = mode(a, loc, scale)
        # FIXME: This loop assumes convergence!
        while True:
            x1 = x0 + (p - cdf(x0, a, loc, scale))/pdf(x0, a, loc, scale)
            if mp.almosteq(x1, x0):
                break
            x0 = x1
        return x1


def sf(x, a, loc=0, scale=1):
    """
    Survival function for the inverse gamma distribution.
    """
    with mp.extradps(5):
        a, loc, scale = _validate_params(a, loc, scale)
        x = mp.mpf(x)
        if x <= loc:
            return mp.one
        z = (x - loc)/scale
        return mp.gammainc(a, 0, 1/z, regularized=True)


def invsf(p, a, loc=0, scale=1):
    """
    Inverse of the survival function for the inverse gamma distribution.
    """
    with mp.extradps(5):
        a, loc, scale = _validate_params(a, loc, scale)
        p = _validate_p(p)
        if p == 0:
            return mp.inf
        if p == 1:
            return loc
        x0 = mode(a, loc, scale)
        # FIXME: This loop assumes convergence!
        while True:
            x1 = x0 - (p - sf(x0, a, loc, scale))/pdf(x0, a, loc, scale)
            if mp.almosteq(x1, x0):
                break
            x0 = x1
        return x1


def mean(a, loc=0, scale=1):
    """
    Mean of the inverse gamma distribution.

    Returns `nan` is `a <= 1`.
    """
    with mp.extradps(5):
        a, loc, scale = _validate_params(a, loc, scale)
        return loc + scale / (a - 1) if a > 1 else mp.nan


def mode(a, loc=0, scale=1):
    """
    Mode of the inverse gamma distribution.
    """
    with mp.extradps(5):
        a, loc, scale = _validate_params(a, loc, scale)
        return loc + scale / (a + 1)


def var(a, loc=0, scale=1):
    """
    Variance of the inverse gamma distribution.

    Returns `nan` if `a <= 2`.
    """
    with mp.extradps(5):
        a, loc, scale = _validate_params(a, loc, scale)
        return scale**2/(a - 1)**2/(a - 2) if a > 2 else mp.nan


def entropy(a, loc=0, scale=1):
    """
    Differential entropy of the inverse gamma distribution.
    """
    with mp.extradps(5):
        a, loc, scale = _validate_params(a, loc, scale)
        return a + mp.log(scale) + mp.loggamma(a) - (1 + a)*mp.digamma(a)


def _standard_noncentral_moment(n, a):
    with mp.extradps(5):
        if n == 0:
            return mp.one
        if a <= n:
            return mp.nan
        return mp.gammaprod([a - n], [a])


def noncentral_moment(n, a, loc=0, scale=1):
    """
    Noncentral moment of the inverse gamma distribution.

    The value is also known as the raw moment.
    """
    # This is a generic calculation that could be applied to any
    # loc/scale family if there is a function for the standard
    # (i.e. loc=0, scale=1) noncentral moment.
    # Cf. genextreme.noncentral_moment()
    with mp.extradps(5):
        n = _validate_moment_n(n)
        a, loc, scale = _validate_params(a, loc, scale)
        if n == 0:
            return mp.one
        terms = [(mp.binomial(n, k) * mp.power(loc, n - k) * mp.power(scale, k)
                  * _standard_noncentral_moment(k, a))
                 for k in range(n + 1)]
        return mp.fsum(terms)
