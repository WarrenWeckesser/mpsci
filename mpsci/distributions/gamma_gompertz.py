"""
Gamma-Gompertz distribution
---------------------------

The distribution is described in:

    https://en.wikipedia.org/wiki/Gamma/Gompertz_distribution

The parameters used here map to the wikipedia article as follows::

    mpsci    wikipedia
    -----    ---------
    c        s
    beta     beta
    scale    1/b

"""

import mpmath
from ._common import _validate_p


__all__ = ['pdf', 'cdf', 'invcdf', 'sf', 'invsf']


def pdf(x, c, beta, scale):
    """
    Probability density function of the Gamma-Gompertz distribution.
    """
    with mpmath.extradps(5):
        if x < 0:
            return mpmath.mp.zero
        x = mpmath.mpf(x)
        beta = mpmath.mpf(beta)
        c = mpmath.mpf(c)
        scale = mpmath.mpf(scale)

        ex = mpmath.exp(x/scale)
        num = c * ex * mpmath.power(beta, c)
        den = scale * mpmath.power(beta - 1 + ex, c + 1)
        return num / den


def cdf(x, c, beta, scale):
    """
    Cumulative distribution function of the Gamma-Gompertz distribution.
    """
    with mpmath.extradps(5):
        if x < 0:
            return mpmath.mp.zero
        x = mpmath.mpf(x)
        beta = mpmath.mpf(beta)
        c = mpmath.mpf(c)
        scale = mpmath.mpf(scale)

        ex = mpmath.exp(x/scale)
        p = -mpmath.powm1(beta / (beta - 1 + ex), c)
        return p


def invcdf(p, c, beta, scale):
    """
    Inverse CDF (i.e. quantile function) of the Gamma-Gompertz distribution.
    """
    with mpmath.extradps(5):
        p = _validate_p(p)
        beta = mpmath.mpf(beta)
        c = mpmath.mpf(c)
        scale = mpmath.mpf(scale)
        # XXX It would be nice if the result could be formulated in a
        # way that avoids computing 1 - p.
        r = mpmath.powm1(1 - p, -1/c)
        x = scale * mpmath.log1p(beta * r)
        return x


def sf(x, c, beta, scale):
    """
    Cumulative distribution function of the Gamma-Gompertz distribution.
    """
    with mpmath.extradps(5):
        if x < 0:
            return mpmath.mp.one
        x = mpmath.mpf(x)
        beta = mpmath.mpf(beta)
        c = mpmath.mpf(c)
        scale = mpmath.mpf(scale)

        ex = mpmath.exp(x/scale)
        p = mpmath.power(beta / (beta - 1 + ex), c)
        return p


def invsf(p, c, beta, scale):
    """
    Inverse survival function of the Gamma-Gompertz distribution.
    """
    with mpmath.extradps(5):
        p = _validate_p(p)
        beta = mpmath.mpf(beta)
        c = mpmath.mpf(c)
        scale = mpmath.mpf(scale)
        r = mpmath.powm1(p, -1/c)
        x = scale * mpmath.log1p(beta * r)
        return x
