"""
Power normal distribution
-------------------------

See https://www.itl.nist.gov/div898/handbook/eda/section3/eda366d.htm

The parameters used here match those used in SciPy.

"""

from mpmath import mp
from mpsci.distributions import normal
from ._common import _validate_p


__all__ = ['pdf', 'logpdf', 'cdf', 'invcdf', 'sf', 'invsf']


def _validate_params(c, loc, scale):
    if c <= 0:
        raise ValueError('c must be greater than 0')
    if scale <= 0:
        raise ValueError('scale must be greater than 0')
    return mp.mpf(c), mp.mpf(loc), mp.mpf(scale)


def pdf(x, c, loc=0, scale=1):
    """
    Probability density function for the power normal distribution.
    """
    with mp.extradps(5):
        c, loc, scale = _validate_params(c, loc, scale)
        x = mp.mpf(x)
        z = (x - loc)/scale
        return c * mp.npdf(z) * mp.ncdf(-z)**(c - 1) / scale


def logpdf(x, c, loc=0, scale=1):
    """
    Logarithm of the PDF for the power normal distribution.
    """
    with mp.extradps(5):
        c, loc, scale = _validate_params(c, loc, scale)
        x = mp.mpf(x)
        z = (x - loc)/scale
        return (mp.log(c)
                + normal.logpdf(z)
                + (c - 1)*mp.log(mp.ncdf(-z))
                - mp.log(scale))


def cdf(x, c, loc=0, scale=1):
    """
    Cumulative distribution function for the power normal distribution.
    """
    with mp.extradps(5):
        c, loc, scale = _validate_params(c, loc, scale)
        x = mp.mpf(x)
        z = (x - loc)/scale
        return -mp.expm1(c*mp.log(mp.ncdf(-z)))


def invcdf(p, c, loc=0, scale=1):
    """
    Inverse of the CDF for the power normal distribution.

    This function is also known as the *quantile function*.
    """
    with mp.extradps(5):
        c, loc, scale = _validate_params(c, loc, scale)
        p = _validate_p(p)
        return loc - scale*normal.invcdf(mp.power(mp.one - p, mp.one/c))


def sf(x, c, loc=0, scale=1):
    """
    Survival function for the power normal distribution.
    """
    with mp.extradps(5):
        c, loc, scale = _validate_params(c, loc, scale)
        x = mp.mpf(x)
        z = (x - loc)/scale
        return mp.exp(c*mp.log(mp.ncdf(-z)))


def invsf(p, c, loc=0, scale=1):
    """
    Inverse of the survival function for the power normal distribution.
    """
    with mp.extradps(5):
        c, loc, scale = _validate_params(c, loc, scale)
        p = _validate_p(p)
        return loc - scale*normal.invcdf(mp.power(p, mp.one/c))
