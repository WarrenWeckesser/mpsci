"""
Chi-square distribution
-----------------------
"""

from mpmath import mp
from ._common import _validate_moment_n, _generic_inv


__all__ = ['pdf', 'logpdf',
           'cdf', 'invcdf', 'invcdf_k', 'sf', 'invsf', 'invsf_k',
           'mean', 'mode', 'var',
           'support', 'entropy', 'noncentral_moment']


def _validate_k(k):
    if k <= 0:
        raise ValueError('k must be positive')
    return mp.mpf(k)


def pdf(x, k):
    """
    PDF for the chi-square distribution.
    """
    with mp.extradps(5):
        k = _validate_k(k)
        x = mp.mpf(x)
        if x < 0:
            return mp.zero
        p = mp.exp(-x/2) * (x/2)**(k/2 - 1)/2 / mp.gamma(k/2)
        return p


def logpdf(x, k):
    """
    Logarithm of the PDF for the chi-square distribution.
    """
    with mp.extradps(5):
        k = _validate_k(k)
        x = mp.mpf(x)
        if x < 0:
            return mp.ninf
        p = -x/2 + (k/2 - 1)*mp.log(x/2) - mp.log(2) - mp.loggamma(k/2)
    return p


def cdf(x, k):
    """
    CDF for the chi-square distribution.
    """
    with mp.extradps(5):
        k = _validate_k(k)
        x = mp.mpf(x)
        if x <= 0:
            return mp.zero
        c = mp.gammainc(k/2, a=0, b=x/2, regularized=True)
    return c


def invcdf(p, k, solver='bisect', **kwargs):
    """
    Inverse of the CDF for the chi-square distribution.

    Also known as the quantile function.

    Additional keyword arguments are passed on to `mp.findroot()`.

    If not given in `kwargs`, this function overrides the default
    `maxsteps` of `mp.findroot` and uses (in effect) 4*mp.prec.

    Experimental!
    """
    return _generic_inv(lambda x: cdf(x, k), p,
                        direction=1, solver=solver, **kwargs)


def invcdf_k(x, p, solver='bisect', **kwargs):
    """
    Inverse with respect to k of the CDF chi2.cdf(x, k).

    Additional keyword arguments are passed on to `mp.findroot()`.

    If not given in `kwargs`, this function overrides the default
    `maxsteps` of `mp.findroot` and uses (in effect) 4*mp.prec.

    Experimental!
    """
    return _generic_inv(lambda k: cdf(x, k), p,
                        direction=-1, solver=solver, **kwargs)


def sf(x, k):
    """
    Survival function for the chi-square distribution.
    """
    with mp.extradps(5):
        k = _validate_k(k)
        x = mp.mpf(x)
        if x <= 0:
            return mp.one
        s = mp.gammainc(k/2, a=x/2, b=mp.inf, regularized=True)
    return s


def invsf(p, k, solver='bisect', **kwargs):
    """
    Inverse of the survival function for the chi-square distribution.

    Also known as the complemented quantile function.

    Additional keyword arguments are passed on to `mp.findroot()`.

    If not given in `kwargs`, this function overrides the default
    `maxsteps` of `mp.findroot` and uses (in effect) 4*mp.prec.

    Experimental!
    """
    return _generic_inv(lambda x: sf(x, k), p,
                        direction=-1, solver=solver, **kwargs)


def invsf_k(x, p, solver='bisect', **kwargs):
    """
    Inverse with respect to k of chi2.sf(x, k).

    Additional keyword arguments are passed on to `mp.findroot()`.

    If not given in `kwargs`, this function overrides the default
    `maxsteps` of `mp.findroot` and uses (in effect) 4*mp.prec.

    Experimental!
    """
    return _generic_inv(lambda k: sf(x, k), p,
                        direction=1, solver=solver, **kwargs)


def support(k):
    """
    Support for the chi-square distribution.
    """
    with mp.extradps(5):
        k = _validate_k(k)
        return (mp.zero, mp.inf)


def mean(k):
    """
    Mean of the chi-square distribution.
    """
    with mp.extradps(5):
        k = _validate_k(k)
        return k


def mode(k):
    """
    Mode of the chi-square distribution.

    The mode is max(k - 2, 0).
    """
    with mp.extradps(5):
        k = _validate_k(k)
        return max(k - 2, mp.zero)


def var(k):
    """
    Variance of the chi-square distribution.
    """
    with mp.extradps(5):
        k = _validate_k(k)
        return 2*k


def entropy(k):
    """
    Differential entropy of the chi-square distribution.
    """
    with mp.extradps(5):
        k = _validate_k(k)
        k2 = k/2
        return k2 + mp.log(2) + mp.loggamma(k2) + (1 - k2)*mp.psi(0, k2)


def noncentral_moment(n, k):
    """
    n-th noncentral moment of the chi-square distribution.

    n must be a nonnegative integer.
    """
    n = _validate_moment_n(n)
    with mp.extradps(5):
        k = _validate_k(k)
        if n == 0:
            return mp.one
        return 2**n * mp.gammaprod([n + k/2], [k/2])
