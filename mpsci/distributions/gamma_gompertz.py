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


__all__ = ['pdf', 'cdf', 'sf']


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
