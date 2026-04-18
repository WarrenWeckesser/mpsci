"""
Cauchy Distribution
-------------------

"""

from mpmath import mp
from ._common import _validate_p, _validate_loc_scale


__all__ = ['support', 'pdf', 'logpdf', 'cdf', 'sf', 'invcdf', 'invsf',
           'mean', 'var', 'entropy']


def support(loc=0, scale=1):
    """
    Support of the Cauchy distribution.
    """
    loc, scale = _validate_loc_scale(loc, scale)
    return (mp.ninf, mp.inf)


@mp.extradps(5)
def pdf(x, loc=0, scale=1):
    """Probability density function of the Cauchy distribution."""
    x = mp.mpf(x)
    loc, scale = _validate_loc_scale(loc, scale)
    z = (x - loc)/scale
    return 1/(mp.pi*scale*(1 + z**2))


@mp.extradps(5)
def logpdf(x, loc=0, scale=1):
    """Natural log of the PDF of the Cauchy distribution."""
    x = mp.mpf(x)
    loc, scale = _validate_loc_scale(loc, scale)
    z = (x - loc)/scale
    return -mp.log(mp.pi) - mp.log(scale) - mp.log1p(z**2)


@mp.extradps(5)
def cdf(x, loc=0, scale=1):
    """Cumulative distribution function of the Cauchy distribution."""
    x = mp.mpf(x)
    loc, scale = _validate_loc_scale(loc, scale)
    z = (x - loc)/scale
    return mp.atan2(1, -z)/mp.pi


@mp.extradps(5)
def invcdf(p, loc=0, scale=1):
    """Quantile function of the Cauchy distribution."""
    p = _validate_p(p)
    loc, scale = _validate_loc_scale(loc, scale)
    return loc - scale*mp.cot(mp.pi*p)


@mp.extradps(5)
def sf(x, loc=0, scale=1):
    """Survival function of the Cauchy distribution."""
    x = mp.mpf(x)
    loc, scale = _validate_loc_scale(loc, scale)
    z = (x - loc)/scale
    return mp.atan2(1, z)/mp.pi


@mp.extradps(5)
def invsf(p, loc=0, scale=1):
    """Inverse of the survival function of the Cauchy distribution."""
    p = _validate_p(p)
    loc, scale = _validate_loc_scale(loc, scale)
    return loc + scale*mp.cot(mp.pi*p)


def mean(loc=0, scale=1):
    """
    Mean of the Cauchy distribution.

    The return value is always nan.
    """
    _validate_loc_scale(loc, scale)
    return mp.nan


def var(loc=0, scale=1):
    """
    Variance of the Cauchy distribution.

    The return value is always nan.
    """
    _validate_loc_scale(loc, scale)
    return mp.nan


@mp.extradps(5)
def entropy(loc=0, scale=1):
    """
    Entropy of the Cauchy distribution.

    The entropy is log(4*pi*scale).
    """
    loc, scale = _validate_loc_scale(loc, scale)
    return mp.log(4*mp.pi*scale)
