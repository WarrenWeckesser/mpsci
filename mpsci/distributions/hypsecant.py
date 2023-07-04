"""
Hyperbolic secant distribution
------------------------------

The wikipedia article

    https://en.wikipedia.org/wiki/Hyperbolic_secant_distribution

provides information about the distribution.  The scale parameter
used here matches that of SciPy's `scipy.special.hypsecant`; it is
a 2/pi times the scale parameter used in the wikipedia article.
"""

from mpmath import mp
from ._common import _validate_p


__all__ = ['pdf', 'logpdf', 'cdf', 'sf', 'invcdf', 'invsf',
           'mean', 'var', 'entropy']


def _validate_loc_scale(loc, scale):
    loc = mp.mpf(loc)
    scale = mp.mpf(scale)
    if scale <= 0:
        raise ValueError('scale must be positive.')
    return loc, scale


def _validate_args(x, loc, scale):
    x = mp.mpf(x)
    loc, scale = _validate_loc_scale(loc, scale)
    return x, loc, scale


def pdf(x, loc=0, scale=1):
    """
    Probability density function for the hyperbolic secant distribution.
    """
    with mp.extradps(5):
        x, loc, scale = _validate_args(x, loc, scale)
        z = (x - loc)/scale
        return mp.sech(z)/mp.pi/scale


def logpdf(x, loc=0, scale=1):
    """
    Logarithm of the PDF of the hyperbolic secant distribution.
    """
    # This could be formulated many ways, but would any of them have
    # numerical advantages?
    return mp.log(pdf(x, loc, scale))


def cdf(x, loc=0, scale=1):
    """
    Cumulative distribution function for the hyperbolic secant distribution.
    """
    with mp.extradps(5):
        x, loc, scale = _validate_args(x, loc, scale)
        z = (x - loc)/scale
        return (2/mp.pi)*mp.atan(mp.exp(z))


def sf(x, loc=0, scale=1):
    """
    Survival function for the hyperbolic secant distribution.
    """
    with mp.extradps(5):
        x, loc, scale = _validate_args(x, loc, scale)
        z = (x - loc)/scale
        return (2/mp.pi)*mp.atan(mp.exp(-z))


def invcdf(p, loc=0, scale=1):
    """
    Inverse of the CDF for the hyperbolic secant distribution.

    This function is also known as the quantile function or the percent
    point function.
    """
    with mp.extradps(5):
        loc, scale = _validate_loc_scale(loc, scale)
        p = _validate_p(p)
        return loc + scale*mp.log(mp.tan(mp.pi*p/2))


def invsf(p, loc=0, scale=1):
    """
    Inverse of the survival function of the hyperblic secant distribution.
    """
    with mp.extradps(5):
        loc, scale = _validate_loc_scale(loc, scale)
        p = _validate_p(p)
        return loc - scale*mp.log(mp.tan(mp.pi*p/2))


def mean(loc=0, scale=1):
    """
    Mean of the hyperbolic secant distribution.
    """
    with mp.extradps(5):
        loc, scale = _validate_loc_scale(loc, scale)
        return loc


def var(loc=0, scale=1):
    """
    Variance of the hyperbolic secant distribution.
    """
    with mp.extradps(5):
        loc, scale = _validate_loc_scale(loc, scale)
        return (mp.pi/2*scale)**2


def entropy(loc=0, scale=1):
    """
    Differential entropy of the hyperbolic secant distribution.
    """
    with mp.extradps(5):
        loc, scale = _validate_loc_scale(loc, scale)
        return mp.log(2*mp.pi) + mp.log(scale)
