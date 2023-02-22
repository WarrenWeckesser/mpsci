"""
Inverse Gaussian distribution
-----------------------------

This implementation uses the same parameterization as the SciPy
implementation in `scipy.stats.invgauss`.  `mu` is a shape parameter;
`loc` and `scale` are the standard location and scale parameters.

"""

from mpmath import mp
from ._common import _validate_p


__all__ = ['pdf', 'logpdf',
           'cdf', 'logcdf', 'invcdf',
           'sf', 'logsf', 'invsf',
           'mean', 'mode', 'var', 'entropy']


def _validate_params(mu, loc, scale):
    if mu <= 0:
        raise ValueError('mu must be positive')
    if scale <= 0:
        raise ValueError('scale must be positive')
    mu = mp.mpf(mu)
    loc = mp.mpf(loc)
    scale = mp.mpf(scale)
    return mu, loc, scale


def pdf(x, mu, loc=0, scale=1):
    """
    PDF for the inverse Gaussian distribution.
    """
    with mp.extradps(5):
        mu, loc, scale = _validate_params(mu, loc, scale)
        x = mp.mpf(x)
        if x <= loc:
            return mp.zero
        z = (x - loc)/scale
        den = scale*mp.sqrt(2*mp.pi*z**3)
        t = ((z - mu)/mu)**2
        num = mp.exp(-t/(2*z))
        return num/den


def logpdf(x, mu, loc=0, scale=1):
    """
    Logarithm of the PDF for the inverse Gaussian distribution.
    """
    with mp.extradps(5):
        mu, loc, scale = _validate_params(mu, loc, scale)
        x = mp.mpf(x)
        if x <= loc:
            return mp.ninf
        z = (x - loc)/scale
        t = ((z - mu)/mu)**2
        logp = (-0.5*mp.log(2*mp.pi) - 1.5*mp.log(z)
                - t/(2*z) - mp.log(scale))
        return logp


def cdf(x, mu, loc=0, scale=1):
    """
    CDF for the inverse Gaussian distribution.
    """
    with mp.extradps(5):
        mu, loc, scale = _validate_params(mu, loc, scale)
        x = mp.mpf(x)
        if x <= loc:
            return mp.zero
        z = (x - loc)/scale
        t1 = mp.ncdf((z/mu - 1)/mp.sqrt(z))
        t2 = mp.exp(2/mu)*mp.ncdf(-(z/mu + 1)/mp.sqrt(z))
        return t1 + t2


def logcdf(x, mu, loc=0, scale=1):
    """
    Logarithm of the CDF for the inverse Gaussian distribution.
    """
    with mp.extradps(5):
        mu, loc, scale = _validate_params(mu, loc, scale)
        x = mp.mpf(x)
        if x <= loc:
            return mp.ninf
        z = (x - loc)/scale
        t1 = mp.log(mp.ncdf((z/mu - 1)/mp.sqrt(z)))
        t2 = (2/mu) + mp.log(mp.ncdf(-(z/mu + 1)/mp.sqrt(z)))
        return t1 + mp.log1p(mp.exp(t2 - t1))


def invcdf(p, mu, loc=0, scale=1):
    """
    Inverse of the CDF for the inverse Gaussian distribution.
    """
    with mp.extradps(5):
        mu, loc, scale = _validate_params(mu, loc, scale)
        p = _validate_p(p)
        if p == 0:
            return loc
        if p == 1:
            return mp.inf
        x0 = mode(mu, loc, scale)
        # FIXME: This loop assumes convergence!
        while True:
            x1 = x0 + (p - cdf(x0, mu, loc, scale))/pdf(x0, mu, loc, scale)
            if mp.almosteq(x1, x0):
                break
            x0 = x1
        return x1


def sf(x, mu, loc=0, scale=1):
    """
    Survival function for the inverse Gaussian distribution.
    """
    with mp.extradps(5):
        mu, loc, scale = _validate_params(mu, loc, scale)
        x = mp.mpf(x)
        if x <= loc:
            return mp.one
        z = (x - loc)/scale
        t1 = mp.ncdf(-(z/mu - 1)/mp.sqrt(z))
        t2 = mp.exp(2/mu)*mp.ncdf(-(z/mu + 1)/mp.sqrt(z))
        return t1 - t2


def logsf(x, mu, loc=0, scale=1):
    """
    Logarithm of the survival function for the inverse Gaussian distribution.
    """
    with mp.extradps(5):
        mu, loc, scale = _validate_params(mu, loc, scale)
        x = mp.mpf(x)
        if x <= loc:
            return mp.zero
        z = (x - loc)/scale
        t1 = mp.log(mp.ncdf(-(z/mu - 1)/mp.sqrt(z)))
        t2 = 2/mu + mp.log(mp.ncdf(-(z/mu + 1)/mp.sqrt(z)))
        return t1 + mp.log1p(-mp.exp(t2 - t1))


def invsf(p, mu, loc=0, scale=1):
    """
    Inverse of the survival function for the inverse Gaussian distribution.
    """
    with mp.extradps(5):
        mu, loc, scale = _validate_params(mu, loc, scale)
        p = _validate_p(p)
        if p == 0:
            return mp.inf
        if p == 1:
            return loc
        x0 = mode(mu, loc, scale)
        # FIXME: This loop assumes convergence!
        while True:
            x1 = x0 - (p - sf(x0, mu, loc, scale))/pdf(x0, mu, loc, scale)
            if mp.almosteq(x1, x0):
                break
            x0 = x1
        return x1


def mean(mu, loc=0, scale=1):
    """
    Mean of the inverse Gaussian distribution.
    """
    with mp.extradps(5):
        mu, loc, scale = _validate_params(mu, loc, scale)
        return scale*mu + loc


def mode(mu, loc=0, scale=1):
    """
    Mode of the inverse Gaussian distribution.
    """
    with mp.extradps(5):
        mu, loc, scale = _validate_params(mu, loc, scale)
        s = 3*mu/2
        # t is equivalent to sqrt(1 + 1/s**2) - 1.
        t = mp.expm1(mp.log1p(1/s**2)/2)
        # m = mu*(sqrt(1 + s**2) - s) = mu*s*(sqrt(1 + 1/s**2) - 1) = mu*s*t
        m = mu*s*t
        return scale*m + loc


def var(mu, loc=0, scale=1):
    """
    Variance of the inverse Gaussian distribution.
    """
    with mp.extradps(5):
        mu, loc, scale = _validate_params(mu, loc, scale)
        return mu**3*scale**2


def entropy(mu, loc=0, scale=1):
    """
    Differential entropy of the inverse Gaussian distribution.
    """
    # Lots of extradps to handle small mu.
    with mp.extradps(5):
        mu, loc, scale = _validate_params(mu, loc, scale)
        t1 = (mp.log(2*mp.pi) + 3*mp.log(mu) + 1)/2
        t2 = 3*(mp.exp(2/mu) * mp.expint(1, 2/mu))/2
        return t1 - t2 + mp.log(scale)
