"""
Gompertz distribution
---------------------

The distribution is described in:

    https://en.wikipedia.org/wiki/Gompertz_distribution

The parameters used here map to the wikipedia article as follows::

    mpsci    wikipedia
    -----    ---------
    c        eta
    scale    1/b

"""

from mpmath import mp
from ._common import _validate_p, _validate_x_bounds


__all__ = ['pdf', 'logpdf', 'cdf', 'sf', 'invcdf', 'invsf', 'mean', 'var',
           'entropy', 'nll']


def _validate_c_scale(c, scale):
    if c <= 0:
        raise ValueError('c must be positive')
    if scale <= 0:
        raise ValueError('scale must be positive')
    return mp.mpf(c), mp.mpf(scale)


def pdf(x, c, scale):
    """
    Probability density function for the Gompertz distribution.

    Note that `scale` is the reciprocal of the `b` parameter in the wikipedia
    article https://en.wikipedia.org/wiki/Gompertz_distribution.
    """
    with mp.extradps(5):
        x = mp.mpf(x)
        c, scale = _validate_c_scale(c, scale)
        if x < 0:
            return mp.zero
        if mp.isinf(x):
            return mp.zero
        z = x/scale
        return c * mp.exp(c + z - c*mp.exp(z)) / scale


def logpdf(x, c, scale):
    """
    Logarithm of th probability density function for the Gompertz distribution.

    Note that `scale` is the reciprocal of the `b` parameter in the wikipedia
    article https://en.wikipedia.org/wiki/Gompertz_distribution.
    """
    with mp.extradps(5):
        x = mp.mpf(x)
        c, scale = _validate_c_scale(c, scale)
        if x < 0:
            return mp.ninf
        if mp.isinf(x):
            return mp.ninf
        z = x/scale
        return mp.log(c) + c + z - c*mp.exp(z) - mp.log(scale)


def cdf(x, c, scale):
    """
    Cumulative distribution function for the Gompertz distribution.

    Note that `scale` is the reciprocal of the `b` parameter in the wikipedia
    article https://en.wikipedia.org/wiki/Gompertz_distribution.
    """
    with mp.extradps(5):
        x = mp.mpf(x)
        c, scale = _validate_c_scale(c, scale)
        if x <= 0:
            return mp.zero
        z = x/scale
        return -mp.expm1(-c*mp.expm1(z))


def invcdf(p, c, scale):
    """
    Inverse of the CDF for the Gompertz distribution.

    Note that `scale` is the reciprocal of the `b` parameter in the wikipedia
    article https://en.wikipedia.org/wiki/Gompertz_distribution.
    """
    with mp.extradps(5):
        p = _validate_p(p)
        c, scale = _validate_c_scale(c, scale)
        if p == 0:
            return mp.zero
        elif p == 1:
            return mp.inf
        return scale*mp.log1p(-mp.log1p(-p)/c)


def sf(x, c, scale):
    """
    Survival function for the Gompertz distribution.

    Note that `scale` is the reciprocal of the `b` parameter in the wikipedia
    article https://en.wikipedia.org/wiki/Gompertz_distribution.
    """
    with mp.extradps(5):
        x = mp.mpf(x)
        c, scale = _validate_c_scale(c, scale)
        if x <= 0:
            return mp.one
        z = x/scale
        return mp.exp(-c*mp.expm1(z))


def invsf(p, c, scale):
    """
    Inverse of the survival function for the Gompertz distribution.

    Note that `scale` is the reciprocal of the `b` parameter in the wikipedia
    article https://en.wikipedia.org/wiki/Gompertz_distribution.
    """
    with mp.extradps(5):
        p = _validate_p(p)
        c, scale = _validate_c_scale(c, scale)
        if p == 0:
            return mp.inf
        elif p == 1:
            return mp.zero
        return scale*mp.log1p(-mp.log(p)/c)


def mean(c, scale):
    """
    Mean of the Gompertz distribution.

    Note that `scale` is the reciprocal of the `b` parameter in the wikipedia
    article https://en.wikipedia.org/wiki/Gompertz_distribution.
    """
    with mp.extradps(5):
        c, scale = _validate_c_scale(c, scale)
        return scale * mp.exp(c) * mp.e1(c)


def _hyp3f3_111222(x):
    x = mp.mpf(x)
    return mp.nsum(lambda k: (-1)**k/(k+1)**3 * x**k/mp.factorial(k),
                   [0, mp.inf])


def var(c, scale):
    """
    Variance of the Gompertz distribution.

    Note that `scale` is the reciprocal of the `b` parameter in the wikipedia
    article https://en.wikipedia.org/wiki/Gompertz_distribution.
    """
    with mp.extradps(5):
        c, scale = _validate_c_scale(c, scale)
        lnc = mp.log(c)
        expc = mp.exp(c)
        return scale**2 * expc * (-2*c*_hyp3f3_111222(c) + mp.euler**2
                                  + mp.pi**2/6 + 2*mp.euler*lnc + lnc**2
                                  - expc*mp.e1(c)**2)


def entropy(c, scale):
    """
    Differential entropy of the Gompertz distribution.

    Note that `scale` is the reciprocal of the `b` parameter in the wikipedia
    article https://en.wikipedia.org/wiki/Gompertz_distribution.
    """
    with mp.extradps(5):
        c, scale = _validate_c_scale(c, scale)
        return mp.one - mp.log(c) - mp.exp(c)*mp.e1(c) + mp.log(scale)


def nll(x, c, scale):
    """
    Negative log-likelihood of a sample for the Gompertz distribution.

    x must be a sequence of numbers.
    """
    with mp.extradps(5):
        c, scale = _validate_c_scale(c, scale)
        x = _validate_x_bounds(x, low=0, high=mp.inf,
                               strict_low=False, strict_high=True)
        return -mp.fsum([logpdf(xi, c, scale) for xi in x])
