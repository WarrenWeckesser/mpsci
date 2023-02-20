"""
Log-logistic distribution
-------------------------

The log-logistic distribution is the probability distribution of a random
variable whose logarithm has a logistic distribution.

The distribution is also known as the Fisk distribution.

The distribution has two parameters, `beta` and `scale`.  `beta` is a
shape parameter.

"""

from mpmath import mp
from ._common import _validate_p


__all__ = ['pdf', 'logpdf', 'cdf', 'sf', 'invcdf', 'invsf', 'mean', 'var']


def _validate_params(beta, scale):
    if beta <= 0:
        raise ValueError('beta must be greater than 0')
    if scale <= 0:
        raise ValueError('scale must be greater than 0')
    return mp.mpf(beta), mp.mpf(scale)


def pdf(x, beta, scale):
    """
    PDF of the log-logistic distribution.
    """
    with mp.extradps(5):
        beta, scale = _validate_params(beta, scale)
        x = mp.mpf(x)
        if x <= 0:
            p = mp.zero
        else:
            z = x / scale
            p = (beta/scale) * z**(beta - 1) / (1 + z**beta)**2
    return p


def logpdf(x, beta, scale):
    """
    Logarithm of the PDF of the log-logistic distribution.
    """
    with mp.extradps(5):
        beta, scale = _validate_params(beta, scale)
        x = mp.mpf(x)
        if x <= 0:
            logp = mp.ninf
        else:
            z = x / scale
            logp = (mp.log(beta) - mp.log(scale) + (beta - 1)*mp.log(z)
                    - 2*mp.log1p(z**beta))
    return logp


def cdf(x, beta, scale):
    """
    CDF of the log-logistic distribution.
    """
    with mp.extradps(5):
        beta, scale = _validate_params(beta, scale)
        x = mp.mpf(x)
        if x <= 0:
            p = mp.zero
        else:
            z = x / scale
            p = 1 / (1 + z**-beta)
    return p


def sf(x, beta, scale):
    """
    Survival function of the log-logistic distribution.
    """
    with mp.extradps(5):
        beta, scale = _validate_params(beta, scale)
        x = mp.mpf(x)
        if x <= 0:
            p = mp.one
        else:
            z = x / scale
            r = z**-beta
            p = r / (1 + r)
    return p


def invcdf(p, beta, scale):
    """
    Inverse CDF of the log-logistic distribution.

    This function is also known as the quantile function or the percent
    point function.
    """
    with mp.extradps(5):
        beta, scale = _validate_params(beta, scale)
        p = _validate_p(p)
        if p == 1:
            return mp.inf
        x = scale * (p / (1 - p))**(1/beta)
    return x


def invsf(p, beta, scale):
    """
    Inverse survival function of the log-logistic distribution.
    """
    with mp.extradps(5):
        beta, scale = _validate_params(beta, scale)
        p = _validate_p(p)
        if p == 0:
            return mp.inf
        x = scale * ((1 - p) / p)**(1/beta)
    return x


def mean(beta, scale):
    """
    Mean of the log-logistic distribution.

    `nan` is returned if beta <= 1.
    """
    with mp.extradps(5):
        beta, scale = _validate_params(beta, scale)
        if beta > 1:
            return scale / mp.sincpi(1/beta)
    return mp.nan


def var(beta, scale):
    """
    Variance of the log-logistic distribution.

    `nan` is returned if beta <= 2.
    """
    with mp.extradps(5):
        beta, scale = _validate_params(beta, scale)
        if beta > 2:
            return scale**2 * (1/mp.sincpi(2/beta)
                               - 1/mp.sincpi(1/beta)**2)
    return mp.nan
