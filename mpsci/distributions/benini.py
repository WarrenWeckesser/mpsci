"""
Benini Distribution
-------------------

The Benini distribution has the probability density function

    f(x, alpha, beta, scale) = exp(theta(x, alpha, beta, scale) * (alpha/x + 2*beta*log(x/scale))/x)

and cumulative distribution function

    F(x, alpha, beta, scale) = 1 - exp(theta(x, alpha, beta, scale)

where

    theta(x, alpha, beta, scale) = -alpha*log(x/scale) - beta*log(x/scale)**2

alpha and beta are shape parameters.

The support is x > scale.

See

* Kleiber, Christian; Kotz, Samuel (2003). "Chapter 7.1: Benini Distribution",
  Statistical Size Distributions in Economics and Actuarial Sciences.
  Wiley. ISBN 978-0-471-15064-0.
* "Benini distribution", Wikipedia,
  https://en.wikipedia.org/wiki/Benini_distribution

"""
import re
from mpmath import mp
from ._common import _validate_p, _validate_x_bounds


# module docstring substitution
_f_expression = r"""
.. math::
        f(x, \\alpha, \\beta, \\sigma) =
            e^{\\theta(x, \\alpha, \\beta, \\sigma)}
            \\left(
                \\frac{\\alpha}{x} +
                \\frac{2\\beta\\log\\left(\\frac{x}{\\sigma}\\right)}{x}
            \\right)
"""

_F_expression = r"""
.. math::
        F(x, \\alpha, \\beta, \\sigma) =
          1 - e^{\\theta(x, \\alpha, \\beta, \\sigma)}
"""

_theta_expression = r"""
.. math::
        \\theta(x, \\alpha, \\beta, \\sigma) =
          -\\alpha \\log\\left(\\frac{x}{\\sigma}\\right)
            - \\beta \\log\\left(\\frac{x}{\\sigma}\\right)^2
"""

_docstring_re_subs = [
    (r'    f\(x,.*$', _f_expression, 0, re.MULTILINE),
    (r'    F\(x,.*$', _F_expression, 0, re.MULTILINE),
    (r'    theta\(x,.*$', _theta_expression, 0, re.MULTILINE),
    (r'alpha and beta are shape parameters.',
     (r':math:`\\alpha` and :math:`\\beta` are shape parameters; '
      r':math:`\\sigma` is a scale parameter.'), 0, 0),
    (r'The support is x > scale.',
     r'The support is :math:`x > \\sigma`.', 0, 0),
]

__all__ = ['support', 'pdf', 'logpdf',
           'cdf', 'logcdf', 'invcdf',
           'sf', 'logsf', 'invsf',
           'median', 'mean', 'var',
           'nll']


def _validate_positive(param, name):
    param = mp.mpf(param)
    if param <= 0:
        raise ValueError(f'{name} must be positive.')
    return param


def _validate_params(alpha, beta, scale):
    alpha = _validate_positive(alpha, 'alpha')
    beta = _validate_positive(beta, 'beta')
    scale = _validate_positive(scale, 'scale')
    return alpha, beta, scale


def support(alpha, beta, scale):
    alpha, beta, scale = _validate_params(alpha, beta, scale)
    return (scale, mp.inf)


@mp.extradps(5)
def logpdf(x, alpha, beta, scale):
    """
    Natural logarithm of the PDF of the Benini distribution.
    """
    alpha, beta, scale = _validate_params(alpha, beta, scale)
    x = mp.mpf(x)
    if x == scale:
        return mp.log(alpha) - mp.log(scale)
    if x < scale:
        return mp.ninf
    logz = mp.log(x/scale)
    blogz = beta*logz
    return -logz*(alpha + blogz) + mp.log(alpha + 2*blogz) - mp.log(x)


@mp.extradps(5)
def pdf(x, alpha, beta, scale):
    """
    Probability density function of the Benini distribution.
    """
    return mp.exp(logpdf(x, alpha, beta, scale))


@mp.extradps(5)
def cdf(x, alpha, beta, scale):
    """
    Cumulative distribution function of the Benini distribution.
    """
    alpha, beta, scale = _validate_params(alpha, beta, scale)
    x = mp.mpf(x)
    if x <= scale:
        return mp.zero
    logz = mp.log(x/scale)
    return -mp.expm1(-logz*(alpha + beta*logz))


@mp.extradps(5)
def logcdf(x, alpha, beta, scale):
    """
    Natural logarithm of the CDF of the Benini distribution.
    """
    alpha, beta, scale = _validate_params(alpha, beta, scale)
    x = mp.mpf(x)
    if x <= scale:
        return mp.ninf
    m = median(alpha, beta, scale)
    if x < m:
        return mp.log(cdf(x, alpha, beta, scale))
    else:
        return mp.log1p(-sf(x, alpha, beta, scale))


@mp.extradps(5)
def invcdf(p, alpha, beta, scale):
    """
    Inverse CDF of the Benini distribution.
    """
    alpha, beta, scale = _validate_params(alpha, beta, scale)
    p = _validate_p(p)
    r = -4*beta/alpha**2 * mp.log1p(-p)
    t = -r/(1 + mp.sqrt(1 + r))
    return scale*mp.exp(-alpha/(2*beta)*t)


@mp.extradps(5)
def sf(x, alpha, beta, scale):
    """
    Cumulative distribution function of the Benini distribution.
    """
    alpha, beta, scale = _validate_params(alpha, beta, scale)
    x = mp.mpf(x)
    if x <= scale:
        return mp.one
    logz = mp.log(x/scale)
    return mp.exp(-logz*(alpha + beta*logz))


@mp.extradps(5)
def logsf(x, alpha, beta, scale):
    """
    Natural logarithm of the survival function of the Benini distribution.
    """
    alpha, beta, scale = _validate_params(alpha, beta, scale)
    x = mp.mpf(x)
    if x <= scale:
        return mp.zero
    m = median(alpha, beta, scale)
    if x < m:
        return mp.log1p(-cdf(x, alpha, beta, scale))
    else:
        logz = mp.log(x/scale)
        return -logz*(alpha + beta*logz)


@mp.extradps(5)
def invsf(p, alpha, beta, scale):
    """
    Inverse survival function of the Benini distribution.
    """
    alpha, beta, scale = _validate_params(alpha, beta, scale)
    p = _validate_p(p)
    r = -4*beta/alpha**2 * mp.log(p)
    t = -r/(1 + mp.sqrt(1 + r))
    return scale*mp.exp(-alpha/(2*beta)*t)


@mp.extradps(5)
def median(alpha, beta, scale):
    return invcdf(0.5, alpha, beta, scale)


@mp.extradps(5)
def _hermite_e(n, x):
    """
    "Probabilist's" Hermite polynomial.
    """
    n = mp.mpf(n)
    x = mp.mpf(x)
    return mp.mpf(2)**(-n/2)*mp.hermite(n, x/mp.sqrt(2))


def h_neg1(x):
    sqrt2 = mp.sqrt(2)
    t = mp.mpf(x)/sqrt2
    return mp.sqrt(mp.pi)/sqrt2 * mp.exp(t**2) * mp.erfc(t)


@mp.extradps(5)
def mean(alpha, beta, scale):
    """
    Mean of the Benini distribution.
    """
    alpha, beta, scale = _validate_params(alpha, beta, scale)
    s = mp.sqrt(2*beta)
    return scale*(1 + h_neg1((alpha - 1)/s)/s)


@mp.extradps(5)
def var(alpha, beta, scale):
    """
    Variance of the Benini distribution.
    """
    alpha, beta, scale = _validate_params(alpha, beta, scale)
    mu = mean(alpha, beta, scale)
    s = mp.sqrt(2*beta)
    g = scale*mp.sqrt(1 + 2*h_neg1((alpha - 2)/s)/s)
    return (g - mu)*(g + mu)


@mp.extradps(5)
def nll(x, alpha, beta, scale):
    """
    Negative log-likelihood for the Benini distribution.

    `x` must be a sequence of numbers, each greater than `scale`.
    """
    with mp.extradps(5):
        alpha, beta, scale = _validate_params(alpha, beta, scale)
        x = _validate_x_bounds(x, low=scale, high=mp.inf)
        return -mp.fsum([logpdf(xi, alpha, beta, scale) for xi in x])
