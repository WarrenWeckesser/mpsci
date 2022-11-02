"""
Inverse Gaussian distribution
-----------------------------

This implementation uses the same parameterization as the SciPy
implementation in `scipy.stats.invgauss`.  `mu` is a shape parameter;
`loc` and `scale` are the standard location and scale parameters.

"""

import mpmath
from ._common import _validate_p


__all__ = ['pdf', 'logpdf',
           'cdf', 'logcdf', 'invcdf',
           'sf', 'logsf', 'invsf',
           'mean', 'mode', 'variance']


def _validate_params(mu, loc, scale):
    if mu <= 0:
        raise ValueError('mu must be positive')
    if scale <= 0:
        raise ValueError('scale must be positive')
    mu = mpmath.mpf(mu)
    loc = mpmath.mpf(loc)
    scale = mpmath.mpf(scale)
    return mu, loc, scale


def pdf(x, mu, loc=0, scale=1):
    """
    PDF for the inverse Gaussian distribution.
    """
    with mpmath.extradps(5):
        mu, loc, scale = _validate_params(mu, loc, scale)
        x = mpmath.mpf(x)
        if x <= loc:
            return mpmath.mp.zero
        z = (x - loc)/scale
        den = mpmath.sqrt(2*mpmath.pi*z**3)
        t = ((z - mu)/mu)**2
        num = mpmath.exp(-t/(2*z))
        return num/den


def logpdf(x, mu, loc=0, scale=1):
    """
    Logarithm of the PDF for the inverse Gaussian distribution.
    """
    with mpmath.extradps(5):
        mu, loc, scale = _validate_params(mu, loc, scale)
        x = mpmath.mpf(x)
        if x <= loc:
            return mpmath.ninf
        z = (x - loc)/scale
        t = ((z - mu)/mu)**2
        logp = (-0.5*mpmath.log(2*mpmath.pi) - 1.5*mpmath.log(z)
                - t/(2*z))
        return logp


def cdf(x, mu, loc=0, scale=1):
    """
    CDF for the inverse Gaussian distribution.
    """
    with mpmath.extradps(5):
        mu, loc, scale = _validate_params(mu, loc, scale)
        x = mpmath.mpf(x)
        if x <= loc:
            return mpmath.mp.zero
        z = (x - loc)/scale
        t1 = mpmath.ncdf((z/mu - 1)/mpmath.sqrt(z))
        t2 = mpmath.exp(2/mu)*mpmath.ncdf(-(z/mu + 1)/mpmath.sqrt(z))
        return t1 + t2


def logcdf(x, mu, loc=0, scale=1):
    """
    Logarithm of the CDF for the inverse Gaussian distribution.
    """
    with mpmath.extradps(5):
        mu, loc, scale = _validate_params(mu, loc, scale)
        x = mpmath.mpf(x)
        if x <= loc:
            return -mpmath.mp.inf
        z = (x - loc)/scale
        t1 = mpmath.log(mpmath.ncdf((z/mu - 1)/mpmath.sqrt(z)))
        t2 = (2/mu) + mpmath.log(mpmath.ncdf(-(z/mu + 1)/mpmath.sqrt(z)))
        return t1 + mpmath.log1p(mpmath.exp(t2 - t1))


def invcdf(p, mu, loc=0, scale=1):
    """
    Inverse of the CDF for the inverse Gaussian distribution.
    """
    with mpmath.extradps(5):
        mu, loc, scale = _validate_params(mu, loc, scale)
        p = _validate_p(p)
        if p == 0:
            return loc
        if p == 1:
            return mpmath.inf
        x0 = mode(mu, loc, scale)
        # FIXME: This loop assumes convergence!
        while True:
            x1 = x0 + (p - cdf(x0, mu, loc, scale))/pdf(x0, mu, loc, scale)
            if mpmath.almosteq(x1, x0):
                break
            x0 = x1
        return x1


def sf(x, mu, loc=0, scale=1):
    """
    Survival function for the inverse Gaussian distribution.
    """
    with mpmath.extradps(5):
        mu, loc, scale = _validate_params(mu, loc, scale)
        x = mpmath.mpf(x)
        if x <= loc:
            return mpmath.mp.one
        z = (x - loc)/scale
        t1 = mpmath.ncdf(-(z/mu - 1)/mpmath.sqrt(z))
        t2 = mpmath.exp(2/mu)*mpmath.ncdf(-(z/mu + 1)/mpmath.sqrt(z))
        return t1 - t2


def logsf(x, mu, loc=0, scale=1):
    """
    Logarithm of the survival function for the inverse Gaussian distribution.
    """
    with mpmath.extradps(5):
        mu, loc, scale = _validate_params(mu, loc, scale)
        x = mpmath.mpf(x)
        if x <= loc:
            return mpmath.mp.zero
        z = (x - loc)/scale
        t1 = mpmath.log(mpmath.ncdf(-(z/mu - 1)/mpmath.sqrt(z)))
        t2 = 2/mu + mpmath.log(mpmath.ncdf(-(z/mu + 1)/mpmath.sqrt(z)))
        return t1 + mpmath.log1p(-mpmath.exp(t2 - t1))


def invsf(p, mu, loc=0, scale=1):
    """
    Inverse of the survival function for the inverse Gaussian distribution.
    """
    with mpmath.extradps(5):
        mu, loc, scale = _validate_params(mu, loc, scale)
        p = _validate_p(p)
        if p == 0:
            return mpmath.inf
        if p == 1:
            return loc
        x0 = mode(mu, loc, scale)
        # FIXME: This loop assumes convergence!
        while True:
            x1 = x0 - (p - sf(x0, mu, loc, scale))/pdf(x0, mu, loc, scale)
            if mpmath.almosteq(x1, x0):
                break
            x0 = x1
        return x1


def mean(mu, loc=0, scale=1):
    """
    Mean of the inverse Gaussian distribution.
    """
    with mpmath.extradps(5):
        mu, loc, scale = _validate_params(mu, loc, scale)
        return scale*mu + loc


def mode(mu, loc=0, scale=1):
    """
    Mode of the inverse Gaussian distribution.
    """
    with mpmath.extradps(5):
        mu, loc, scale = _validate_params(mu, loc, scale)
        s = 3*mu/2
        # t is equivalent to sqrt(1 + 1/s**2) - 1.
        t = mpmath.expm1(mpmath.log1p(1/s**2)/2)
        # m = mu*(sqrt(1 + s**2) - s) = mu*s*(sqrt(1 + 1/s**2) - 1) = mu*s*t
        m = mu*s*t
        return scale*m + loc


def variance(mu, loc=0, scale=1):
    """
    Variance of the inverse Gaussian distribution.
    """
    with mpmath.extradps(5):
        mu, loc, scale = _validate_params(mu, loc, scale)
        return mu**3*scale**2
