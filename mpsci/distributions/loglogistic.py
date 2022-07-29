"""
Log-logistic distribution
-------------------------

The log-logistic distribution is the probability distribution of a random
variable whose logarithm has a logistic distribution.

The distribution has two parameters, `beta` and `scale`.  `beta` is a
shape parameter.

"""

import mpmath


__all__ = ['pdf', 'logpdf', 'cdf', 'sf', 'invcdf', 'invsf', 'mean', 'var']


def _validate_params(beta, scale):
    if beta <= 0:
        raise ValueError('beta must be greater than 0')
    if scale <= 0:
        raise ValueError('scale must be greater than 0')


def _validate_p(p):
    if p < 0 or p > 1:
        raise ValueError('p must be in the interval [0, 1].')


def pdf(x, beta, scale):
    """
    PDF of the log-logistic distribution.
    """
    _validate_params(beta, scale)
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        if x <= 0:
            p = mpmath.mp.zero
        else:
            beta = mpmath.mpf(beta)
            scale = mpmath.mpf(scale)
            z = x / scale
            p = (beta/scale) * z**(beta - 1) / (1 + z**beta)**2
    return p


def logpdf(x, beta, scale):
    """
    Logarithm of the PDF of the log-logistic distribution.
    """
    _validate_params(beta, scale)
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        if x <= 0:
            logp = mpmath.mp.ninf
        else:
            beta = mpmath.mpf(beta)
            scale = mpmath.mpf(scale)
            z = x / scale
            logp = (mpmath.log(beta) - mpmath.log(scale)
                    + (beta - 1)*mpmath.log(z)
                    - 2*mpmath.log1p(z**beta))
    return logp


def cdf(x, beta, scale):
    """
    CDF of the log-logistic distribution.
    """
    _validate_params(beta, scale)
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        if x <= 0:
            p = mpmath.mp.zero
        else:
            beta = mpmath.mpf(beta)
            scale = mpmath.mpf(scale)
            z = x / scale
            p = 1 / (1 + z**-beta)
    return p


def sf(x, beta, scale):
    """
    Survival function of the log-logistic distribution.
    """
    _validate_params(beta, scale)
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        if x <= 0:
            p = mpmath.mp.one
        else:
            beta = mpmath.mpf(beta)
            scale = mpmath.mpf(scale)
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
    _validate_p(p)
    _validate_params(beta, scale)
    with mpmath.extradps(5):
        p = mpmath.mpf(p)
        if p == 1:
            return mpmath.inf
        beta = mpmath.mpf(beta)
        scale = mpmath.mpf(scale)
        x = scale * (p / (1 - p))**(1/beta)
    return x


def invsf(p, beta, scale):
    """
    Inverse survival function of the log-logistic distribution.
    """
    _validate_p(p)
    _validate_params(beta, scale)
    with mpmath.extradps(5):
        p = mpmath.mpf(p)
        if p == 0:
            return mpmath.inf
        beta = mpmath.mpf(beta)
        scale = mpmath.mpf(scale)
        x = scale * ((1 - p) / p)**(1/beta)
    return x


def mean(beta, scale):
    """
    Mean of the log-logistic distribution.

    `nan` is returned if beta <= 1.
    """
    _validate_params(beta, scale)
    with mpmath.extradps(5):
        beta = mpmath.mpf(beta)
        scale = mpmath.mpf(scale)
        if beta > 1:
            return scale / mpmath.sincpi(1/beta)
    return mpmath.nan


def var(beta, scale):
    """
    Variance of the log-logistic distribution.

    `nan` is returned if beta <= 2.
    """
    _validate_params(beta, scale)
    with mpmath.extradps(5):
        beta = mpmath.mpf(beta)
        scale = mpmath.mpf(scale)
        if beta > 2:
            return scale**2 * (1/mpmath.sincpi(2/beta)
                               - 1/mpmath.sincpi(1/beta)**2)
    return mpmath.nan
