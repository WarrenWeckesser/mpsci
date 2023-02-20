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

from mpmath import mp
from ._common import _validate_p


__all__ = ['pdf', 'cdf', 'invcdf', 'sf', 'invsf']


def pdf(x, c, beta, scale):
    """
    Probability density function of the Gamma-Gompertz distribution.
    """
    with mp.extradps(5):
        if x < 0:
            return mp.zero
        x = mp.mpf(x)
        beta = mp.mpf(beta)
        c = mp.mpf(c)
        scale = mp.mpf(scale)

        ex = mp.exp(x/scale)
        num = c * ex * mp.power(beta, c)
        den = scale * mp.power(beta - 1 + ex, c + 1)
        return num / den


def cdf(x, c, beta, scale):
    """
    Cumulative distribution function of the Gamma-Gompertz distribution.
    """
    with mp.extradps(5):
        if x < 0:
            return mp.zero
        x = mp.mpf(x)
        beta = mp.mpf(beta)
        c = mp.mpf(c)
        scale = mp.mpf(scale)

        ex = mp.exp(x/scale)
        p = -mp.powm1(beta / (beta - 1 + ex), c)
        return p


def invcdf(p, c, beta, scale):
    """
    Inverse CDF (i.e. quantile function) of the Gamma-Gompertz distribution.
    """
    with mp.extradps(5):
        p = _validate_p(p)
        beta = mp.mpf(beta)
        c = mp.mpf(c)
        scale = mp.mpf(scale)
        # XXX It would be nice if the result could be formulated in a
        # way that avoids computing 1 - p.
        r = mp.powm1(1 - p, -1/c)
        x = scale * mp.log1p(beta * r)
        return x


def sf(x, c, beta, scale):
    """
    Cumulative distribution function of the Gamma-Gompertz distribution.
    """
    with mp.extradps(5):
        if x < 0:
            return mp.one
        x = mp.mpf(x)
        beta = mp.mpf(beta)
        c = mp.mpf(c)
        scale = mp.mpf(scale)

        ex = mp.exp(x/scale)
        p = mp.power(beta / (beta - 1 + ex), c)
        return p


def invsf(p, c, beta, scale):
    """
    Inverse survival function of the Gamma-Gompertz distribution.
    """
    with mp.extradps(5):
        p = _validate_p(p)
        beta = mp.mpf(beta)
        c = mp.mpf(c)
        scale = mp.mpf(scale)
        r = mp.powm1(p, -1/c)
        x = scale * mp.log1p(beta * r)
        return x
