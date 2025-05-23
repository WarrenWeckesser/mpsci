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
    with mp.extradps(5):
        loc, scale = _validate_loc_scale(loc, scale)
        return (mp.ninf, mp.inf)


def pdf(x, loc=0, scale=1):
    """Probability density function of the Cauchy distribution."""
    with mp.extradps(5):
        x = mp.mpf(x)
        loc, scale = _validate_loc_scale(loc, scale)
        z = (x - loc)/scale
        return 1/(mp.pi*scale*(1 + z**2))


def logpdf(x, loc=0, scale=1):
    """Natural log of the PDF of the Cauchy distribution."""
    with mp.extradps(5):
        x = mp.mpf(x)
        loc, scale = _validate_loc_scale(loc, scale)
        z = (x - loc)/scale
        return -mp.log(mp.pi) - mp.log(scale) - mp.log1p(z**2)


def cdf(x, loc=0, scale=1):
    """Cumulative distribution function of the Cauchy distribution."""
    with mp.extradps(5):
        x = mp.mpf(x)
        loc, scale = _validate_loc_scale(loc, scale)
        z = (x - loc)/scale
        return mp.atan2(1, -z)/mp.pi


def invcdf(p, loc=0, scale=1):
    """Quantile function of the Cauchy distribution."""
    with mp.extradps(5):
        p = _validate_p(p)
        loc, scale = _validate_loc_scale(loc, scale)
        return loc - scale*mp.cot(mp.pi*p)


def sf(x, loc=0, scale=1):
    """Survival function of the Cauchy distribution."""
    with mp.extradps(5):
        x = mp.mpf(x)
        loc, scale = _validate_loc_scale(loc, scale)
        z = (x - loc)/scale
        return mp.atan2(1, z)/mp.pi


def invsf(p, loc=0, scale=1):
    """Inverse of the survival function of the Cauchy distribution."""
    with mp.extradps(5):
        p = _validate_p(p)
        loc, scale = _validate_loc_scale(loc, scale)
        return loc + scale*mp.cot(mp.pi*p)


def mean(loc=0, scale=1):
    """
    Mean of the Cauchy distribution.

    The return value is always nan.
    """
    with mp.extradps(5):
        _validate_loc_scale(loc, scale)
        return mp.nan


def var(loc=0, scale=1):
    """
    Variance of the Cauchy distribution.

    The return value is always nan.
    """
    with mp.extradps(5):
        _validate_loc_scale(loc, scale)
        return mp.nan


def entropy(loc=0, scale=1):
    """
    Entropy of the Cauchy distribution.

    The entropy is log(4*pi*scale).
    """
    with mp.extradps(5):
        loc, scale = _validate_loc_scale(loc, scale)
        return mp.log(4*mp.pi*scale)
