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

import mpmath
from ._common import _validate_p


__all__ = ['pdf', 'logpdf', 'cdf', 'sf', 'invcdf', 'invsf', 'mean']


def _validate_c_scale(c, scale):
    if c <= 0:
        raise ValueError('c must be positive')
    if scale <= 0:
        raise ValueError('scale must be positive')
    return mpmath.mpf(c), mpmath.mpf(scale)


def pdf(x, c, scale):
    """
    Probability density function for the Gompertz distribution.

    Note that `scale` is the reciprocal of the `b` parameter in the wikipedia
    article https://en.wikipedia.org/wiki/Gompertz_distribution.
    """
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        c, scale = _validate_c_scale(c, scale)
        if x < 0:
            return mpmath.mp.zero
        if mpmath.isinf(x):
            return mpmath.mp.zero
        z = x/scale
        return c * mpmath.exp(c + z - c*mpmath.exp(z)) / scale


def logpdf(x, c, scale):
    """
    Logarithm of th probability density function for the Gompertz distribution.

    Note that `scale` is the reciprocal of the `b` parameter in the wikipedia
    article https://en.wikipedia.org/wiki/Gompertz_distribution.
    """
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        c, scale = _validate_c_scale(c, scale)
        if x < 0:
            return mpmath.ninf
        if mpmath.isinf(x):
            return mpmath.ninf
        z = x/scale
        return mpmath.log(c) + c + z - c*mpmath.exp(z) - mpmath.log(scale)


def cdf(x, c, scale):
    """
    Cumulative distribution function for the Gompertz distribution.

    Note that `scale` is the reciprocal of the `b` parameter in the wikipedia
    article https://en.wikipedia.org/wiki/Gompertz_distribution.
    """
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        c, scale = _validate_c_scale(c, scale)
        if x <= 0:
            return mpmath.mp.zero
        z = x/scale
        return -mpmath.expm1(-c*mpmath.expm1(z))


def invcdf(p, c, scale):
    """
    Inverse of the CDF for the Gompertz distribution.

    Note that `scale` is the reciprocal of the `b` parameter in the wikipedia
    article https://en.wikipedia.org/wiki/Gompertz_distribution.
    """
    with mpmath.extradps(5):
        p = _validate_p(p)
        c, scale = _validate_c_scale(c, scale)
        if p == 0:
            return mpmath.mp.zero
        elif p == 1:
            return mpmath.inf
        return scale*mpmath.log1p(-mpmath.log1p(-p)/c)


def sf(x, c, scale):
    """
    Survival function for the Gompertz distribution.

    Note that `scale` is the reciprocal of the `b` parameter in the wikipedia
    article https://en.wikipedia.org/wiki/Gompertz_distribution.
    """
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        c, scale = _validate_c_scale(c, scale)
        if x <= 0:
            return mpmath.mp.one
        z = x/scale
        return mpmath.exp(-c*mpmath.expm1(z))


def invsf(p, c, scale):
    """
    Inverse of the survival function for the Gompertz distribution.

    Note that `scale` is the reciprocal of the `b` parameter in the wikipedia
    article https://en.wikipedia.org/wiki/Gompertz_distribution.
    """
    with mpmath.extradps(5):
        p = _validate_p(p)
        c, scale = _validate_c_scale(c, scale)
        if p == 0:
            return mpmath.inf
        elif p == 1:
            return mpmath.mp.zero
        return scale*mpmath.log1p(-mpmath.log(p)/c)


def mean(c, scale):
    """
    Mean of the Gompertz distribution.

    Note that `scale` is the reciprocal of the `b` parameter in the wikipedia
    article https://en.wikipedia.org/wiki/Gompertz_distribution.
    """
    with mpmath.extradps(5):
        c, scale = _validate_c_scale(c, scale)
        return scale * mpmath.exp(c) * mpmath.e1(c)
