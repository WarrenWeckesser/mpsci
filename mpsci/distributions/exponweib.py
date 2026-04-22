"""
Exponentiated Weibull probability distribution
----------------------------------------------
"""

from mpmath import mp
from ._common import _validate_p

__all__ = ['pdf', 'logpdf', 'cdf', 'invcdf', 'sf', 'invsf',
           'support', 'noncentral_moment', 'mean', 'var']


def _validate_params(a, c, scale):
    if a <= 0:
        raise ValueError('`a` must be greater than 0')
    if c <= 0:
        raise ValueError('`c` must be greater than 0')
    if scale <= 0:
        raise ValueError('`scale` must be greater than 0')
    return mp.mpf(a), mp.mpf(c), mp.mpf(scale)


@mp.extradps(5)
def pdf(x, a, c, scale=1):
    """
    PDF for the exponentiated Weibull distribution.

    All the distribution parameters are assumed to be positive.
    """
    a, c, scale = _validate_params(a, c, scale)
    x = mp.mpf(x)
    if x < 0:
        return mp.zero
    z = x/scale
    p = (a * c / scale * z**(c-1) * (-mp.expm1(-z**c))**(a - 1) *
         mp.exp(-z**c))
    return p


@mp.extradps(5)
def logpdf(x, a, c, scale=1):
    """
    Logarithm of the PDF for the exponentiated Weibull distribution.

    All the distribution parameters are assumed to be positive.
    """
    a, c, scale = _validate_params(a, c, scale)
    x = mp.mpf(x)
    if x < 0:
        return mp.ninf
    z = x/scale
    logp = (mp.log(a)
            + mp.log(c)
            - mp.log(scale)
            + (c - 1)*mp.log(z)
            + (a - 1)*mp.log(-mp.expm1(-z**c))
            - z**c)
    return logp


@mp.extradps(5)
def cdf(x, a, c, scale=1):
    """
    CDF for the exponentiated Weibull distribution.

    This is the cumulative distribution function for the exponentiated
    Weibull distribution.

    All the distribution parameters are assumed to be positive.
    """
    a, c, scale = _validate_params(a, c, scale)
    x = mp.mpf(x)
    if x < 0:
        return mp.zero
    z = x/scale
    return mp.power(-mp.expm1(-z**c), a)


@mp.extradps(5)
def invcdf(p, a, c, scale=1):
    """
    Inverse of the CDF of the exponentiated Weibull distribution.
    """
    a, c, scale = _validate_params(a, c, scale)
    p = _validate_p(p)
    return scale*(-mp.log1p(-p**(1/a)))**(1/c)


@mp.extradps(5)
def sf(x, a, c, scale=1):
    """
    Survival function of the exponentiated Weibull distribution.
    """
    a, c, scale = _validate_params(a, c, scale)
    x = mp.mpf(x)
    if x < 0:
        return mp.one
    z = x/scale
    return -mp.powm1(-mp.expm1(-z**c), a)


@mp.extradps(5)
def invsf(p, a, c, scale=1):
    """
    Inverse of the survival function of the exponentiated Weibull distribution.
    """
    a, c, scale = _validate_params(a, c, scale)
    p = _validate_p(p)
    return scale*(-mp.log(-mp.expm1(mp.log1p(-p)/a)))**(1/c)


def support(a, c, scale=1):
    """
    Support of the exponentiated Weibull distribution.
    """
    a, c, scale = _validate_params(a, c, scale)
    return (mp.zero, mp.inf)


def _a_coeff(i, theta):
    """
    See equation (3.2) of [1].

    The parameter name `theta` is from the paper; it is the same as the `a`
    parameter of the functions defined in this module. `theta` is used here
    to match the paper and avoid confusion with the name of the `a` coefficient
    as used in the paper.
    """
    y = 1
    for j in range(i):
        y *= (theta - 1 - j)
    return mp.power(-1, i) * y / mp.factorial(i)


def _term(i, k, theta, c):
    """
    Compute the i-th term of the infinite series in equation (3.3) of [1].

    The parameter name `theta` is from the paper; it is the same as the `a`
    parameter of the functions defined in this module. `theta` is used here
    to match the paper and avoid confusion with the name of the `a` coefficient
    as used in the paper.
    """
    i = int(i)
    c = mp.mpf(c)
    return _a_coeff(i, theta) * mp.power(i + 1, -(k/c + 1))


@mp.extradps(5)
def noncentral_moment(k, a, c, scale=1):
    """
    Compute the raw moment of the exponentiated Weibull distribution.

    The function uses mpmath.nsum to implement equation (3.3) of the paper [1].

    ..[1] Choudhury, A. (2005). "A Simple Derivation of Moments of the Exponentiated
          Weibull Distribution". Metrika. 62 (1): 17-22. doi:10.1007/s001840400351
    """
    a, c, scale = _validate_params(a, c, scale)
    s = mp.nsum(lambda i: _term(i, k, a, c), [1, mp.inf], steps=[100, 10], method='l')
    return a * mp.power(scale, k) * mp.gamma(k/c + 1) * (1 + s)


def mean(a, c, scale=1):
    """
    Mean of the exponentiated Weibull distribution.

    This is a simple wrapper around `noncentral_moment(1, a, c, scale)`.
    """
    return noncentral_moment(1, a, c, scale)


@mp.extradps(5)
def var(a, c, scale=1):
    """
    Variance of the exponentiated Weibull distribution.

    This function computes

        noncentral_moment(2, a, c, scale) - noncentral_moment(1, a, c, scale)**2

    Increase the precision if there is a possibiliy of loss of precision in
    the subtraction.
    """
    m1 = noncentral_moment(1, a, c, scale)
    m2 = noncentral_moment(2, a, c, scale)
    return m2 - m1**2
