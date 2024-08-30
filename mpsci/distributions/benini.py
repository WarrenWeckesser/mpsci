"""
Benini Distribution
-------------------

See

* Kleiber, Christian; Kotz, Samuel (2003). "Chapter 7.1: Benini Distribution",
  Statistical Size Distributions in Economics and Actuarial Sciences.
  Wiley. ISBN 978-0-471-15064-0.
* "Benini distribution", Wikipedia,
  https://en.wikipedia.org/wiki/Benini_distribution

"""
from mpmath import mp
from ._common import _validate_p


__all__ = ['support', 'pdf', 'logpdf', 'cdf', 'invcdf', 'sf', 'invsf',
           'mean', 'var']


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
    if x <= scale:
        return mp.zero
    logz = mp.log(x/scale)
    return -mp.expm1(-logz*(alpha + beta*logz))


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
    if x <= scale:
        return mp.one
    logz = mp.log(x/scale)
    return mp.exp(-logz*(alpha + beta*logz))


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
