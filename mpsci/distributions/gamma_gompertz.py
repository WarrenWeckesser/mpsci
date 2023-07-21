"""
Gamma/Gompertz distribution
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
from ..fun import pow1pm1


__all__ = ['pdf', 'logpdf', 'cdf', 'invcdf', 'sf', 'invsf', 'mean']


def _validate_params(c, beta, scale):
    if c <= 0:
        raise ValueError('c must be greater than 0')
    if beta <= 0:
        raise ValueError('beta must be greater than 0')
    if scale <= 0:
        raise ValueError('scale must be greater than 0')
    return mp.mpf(c), mp.mpf(beta), mp.mpf(scale)


def pdf(x, c, beta, scale):
    """
    Probability density function of the Gamma/Gompertz distribution.
    """
    with mp.extradps(5):
        c, beta, scale = _validate_params(c, beta, scale)
        x = mp.mpf(x)
        if x < 0:
            return mp.zero
        z = x/scale
        num = c * mp.exp(z) * mp.power(beta, c)
        den = scale * mp.power(beta + mp.expm1(z), c + 1)
        return num / den


def logpdf(x, c, beta, scale):
    """
    Probability density function of the Gamma/Gompertz distribution.
    """
    with mp.extradps(5):
        c, beta, scale = _validate_params(c, beta, scale)
        x = mp.mpf(x)
        if x < 0:
            return mp.ninf
        z = x/scale
        return (mp.log(c) + z + c*mp.log(beta)
                - (c + 1)*mp.log(beta + mp.expm1(z)) - mp.log(scale))


def cdf(x, c, beta, scale):
    """
    Cumulative distribution function of the Gamma/Gompertz distribution.
    """
    with mp.extradps(5):
        c, beta, scale = _validate_params(c, beta, scale)
        x = mp.mpf(x)
        if x < 0:
            return mp.zero
        z = x/scale
        if beta == 1:
            return -mp.expm1(-c*z)
        else:
            p = -mp.powm1(beta / (beta + mp.expm1(z)), c)
            return p


def invcdf(p, c, beta, scale):
    """
    Inverse CDF (i.e. quantile function) of the Gamma/Gompertz distribution.
    """
    with mp.extradps(5):
        c, beta, scale = _validate_params(c, beta, scale)
        p = _validate_p(p)
        r = pow1pm1(-p, -1/c)
        x = scale * mp.log1p(beta * r)
        return x


def sf(x, c, beta, scale):
    """
    Cumulative distribution function of the Gamma/Gompertz distribution.
    """
    with mp.extradps(5):
        c, beta, scale = _validate_params(c, beta, scale)
        x = mp.mpf(x)
        if x < 0:
            return mp.one
        z = x/scale
        if beta == 1:
            return mp.exp(-c*z)
        else:
            ex = mp.exp(z)
            p = mp.power(beta / (beta - 1 + ex), c)
            return p


def invsf(p, c, beta, scale):
    """
    Inverse survival function of the Gamma/Gompertz distribution.
    """
    with mp.extradps(5):
        c, beta, scale = _validate_params(c, beta, scale)
        p = _validate_p(p)
        r = mp.powm1(p, -1/c)
        x = scale * mp.log1p(beta * r)
        return x


def mean(c, beta, scale):
    """
    Mean of the Gamma/Gompertz distribution.
    """
    with mp.extradps(5):
        c, beta, scale = _validate_params(c, beta, scale)
        if beta == 1:
            return scale/c
        if c == 1:
            return scale * beta/(beta - 1) * mp.log(beta)
        return scale/c * mp.hyp2f1(c, 1, c+1, (beta - 1)/beta)
