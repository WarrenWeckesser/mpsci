"""
Maxwell distribution
--------------------

The Maxwell distribution is also known as the Maxwell-Boltzmann distribution.
See https://en.wikipedia.org/wiki/Maxwell%E2%80%93Boltzmann_distribution.

The parametrization used here matches that of SciPy.

* ``loc`` is the location parameter.  (The wikipedia article does not
  include a location parameter for the distribution.)
* ``scale`` is the scale parameter. ``scale`` is the parameter ``a``
  in the wikipedia article.

"""

from mpmath import mp
from ._common import _validate_p, _find_bracket


__all__ = ['pdf', 'logpdf', 'cdf', 'invcdf', 'sf', 'invsf',
           'support', 'mean', 'mode', 'var', 'entropy']


def _validate_params(loc=0, scale=1):
    if scale <= 0:
        raise ValueError('scale must be positive')
    return mp.mpf(loc), mp.mpf(scale)


def pdf(x, loc=0, scale=1):
    """
    Probability density function for the Maxwell distribution.
    """
    with mp.extradps(5):
        loc, scale = _validate_params(loc, scale)
        if x <= loc:
            return mp.zero
        x = mp.mpf(x)
        z = (x - loc)/scale
        z2 = z**2
        return mp.sqrt(2/mp.pi)*z2*mp.exp(-z2/2)/scale


def logpdf(x, loc=0, scale=1):
    """
    Natural logarithm of the PDF of the Maxwell distribution.
    """
    with mp.extradps(5):
        loc, scale = _validate_params(loc, scale)
        if x <= loc:
            return mp.ninf
        x = mp.mpf(x)
        z = (x - loc)/scale
        return mp.log(2/mp.pi)/2 + 2*mp.log(z) - z**2/2 - mp.log(scale)


def cdf(x, loc=0, scale=1):
    """
    Cumulative distribution function for the Maxwell distribution.
    """
    with mp.extradps(5):
        loc, scale = _validate_params(loc, scale)
        if x <= loc:
            return mp.zero
        x = mp.mpf(x)
        z = (x - loc)/scale
        return mp.gammainc(3/2, 0, z**2/2, regularized=True)


def invcdf(p, loc=0, scale=1):
    """
    Inverse of the CDF of the Maxwell distribution.

    This function is also known as the quantile function.

    The function is implemented by numerically inverting the CDF
    using the mpmath ``findroot`` function.  It may fail for
    extremely small values of ``p``.
    """
    with mp.extradps(5):
        p = _validate_p(p)
        loc, scale = _validate_params(loc, scale)
        if p == 0:
            return mp.zero
        if p == 1:
            return mp.inf
        x0, x1 = _find_bracket(lambda x: cdf(x, loc, scale), p, 0, mp.inf)
        if x0 == x1:
            return x0
        try:
            x = mp.findroot(lambda x: cdf(x, loc, scale) - p, x0=(x0, x1),
                            solver='secant')
        except Exception:
            x = mp.findroot(lambda x: cdf(x, loc, scale) - p, x0=(x0 + x1)/2,
                            solver='newton')
        return x


def sf(x, loc=0, scale=1):
    """
    Survival function for the Maxwell distribution.
    """
    with mp.extradps(5):
        loc, scale = _validate_params(loc, scale)
        if x <= loc:
            return mp.one
        x = mp.mpf(x)
        z = (x - loc)/scale
        return mp.gammainc(3/2, z**2/2, mp.inf, regularized=True)


def invsf(p, loc=0, scale=1):
    """
    Inverse of the survival function of the Maxwell distribution.

    The function is implemented by numerically inverting the
    survival function using the mpmath ``findroot`` function.
    It may fail for values of ``p`` very close to 1.
    """
    with mp.extradps(5):
        p = _validate_p(p)
        loc, scale = _validate_params(loc, scale)
        if p == 0:
            return mp.inf
        if p == 1:
            return mp.zero
        x0, x1 = _find_bracket(lambda x: sf(x, loc, scale), p, 0, mp.inf)
        if x0 == x1:
            return x0
        try:
            x = mp.findroot(lambda x: sf(x, loc, scale) - p, x0=(x0, x1),
                            solver='secant')
        except Exception:
            x = mp.findroot(lambda x: cdf(x, loc, scale) - p, x0=(x0 + x1)/2,
                            solver='newton')
        return x


def support(loc=0, scale=1):
    """
    Support of the Maxwell distribution.
    """
    with mp.extradps(5):
        loc, scale = _validate_params(loc, scale)
        return (loc, mp.inf)


def mean(loc=0, scale=1):
    """
    Mean of the Maxwell distribution.
    """
    with mp.extradps(5):
        loc, scale = _validate_params(loc, scale)
        return loc + scale*2*mp.sqrt(2/mp.pi)


def mode(loc=0, scale=1):
    """
    Mean of the Maxwell distribution.
    """
    with mp.extradps(5):
        loc, scale = _validate_params(loc, scale)
        return loc + scale*mp.sqrt(2)


def var(loc=0, scale=1):
    """
    Variance of the Maxwell distribution.
    """
    with mp.extradps(5):
        loc, scale = _validate_params(loc, scale)
        return scale**2 * (3*mp.pi - 8)/mp.pi


def entropy(loc=0, scale=1):
    """
    Differential entropy of the Maxwell distribution.
    """
    with mp.extradps(5):
        loc, scale = _validate_params(loc, scale)
        return mp.log(scale*mp.sqrt(2*mp.pi)) + mp.euler - mp.one/2
