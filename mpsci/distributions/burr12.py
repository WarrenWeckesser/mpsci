"""
Burr type XII probability distribution
--------------------------------------

"""

from mpmath import mp
from mpsci.distributions._common import _validate_p


__all__ = ['pdf', 'logpdf', 'cdf', 'invcdf', 'sf', 'invsf', 'logsf',
           'mean', 'var', 'median', 'mode']


def _validate_params(c, d, scale):
    if c <= 0:
        raise ValueError('c must be greater than 0.')
    if d <= 0:
        raise ValueError('d must be greater than 0.')
    if scale <= 0:
        raise ValueError('scale must be greater than 0.')
    return mp.mpf(c), mp.mpf(d), mp.mpf(scale)


def pdf(x, c, d, scale):
    """
    Probability density function of the Burr type XII distribution.
    """
    with mp.extradps(5):
        c, d, scale = _validate_params(c, d, scale)
        x = mp.mpf(x)
        if x <= 0:
            return mp.zero
        z = x/scale
        return c*d*z**(c - 1)/scale / (1 + z**c)**(d+1)


def logpdf(x, c, d, scale):
    """
    Log of the PDF of the Burr type XII distribution.
    """
    with mp.extradps(5):
        c, d, scale = _validate_params(c, d, scale)
        x = mp.mpf(x)
        if x <= 0:
            return mp.ninf
        return (mp.log(c) + mp.log(d) + (c - 1)*mp.log(x)
                - c*mp.log(scale) - (d + 1)*mp.log1p((x / scale)**c))


def cdf(x, c, d, scale):
    """
    Burr type XII distribution cumulative distribution function.
    """
    with mp.extradps(5):
        c, d, scale = _validate_params(c, d, scale)
        x = mp.mpf(x)
        if x <= 0:
            return mp.zero
        # TO DO: See if the use of logsf (as in scipy) is worthwhile.
        return 1 - sf(x, c, d, scale)


def invcdf(p, c, d, scale):
    """
    Inverse of the CDF of the Burr type XII distribution.
    """
    with mp.extradps(5):
        c, d, scale = _validate_params(c, d, scale)
        p = _validate_p(p)
        return scale * mp.powm1(1 - p, -1/d)**(1/c)


def sf(x, c, d, scale):
    """
    Survival function of the Burr type XII distribution.
    """
    with mp.extradps(5):
        c, d, scale = _validate_params(c, d, scale)
        x = mp.mpf(x)
        if x <= 0:
            return mp.one
        return (1 + (x/scale)**c)**(-d)


def invsf(p, c, d, scale):
    """
    Inverse of the survival function of the Burr type XII distribution.
    """
    with mp.extradps(5):
        c, d, scale = _validate_params(c, d, scale)
        p = _validate_p(p)
        return scale * mp.powm1(p, -1/d)**(1/c)


def logsf(x, c, d, scale):
    """
    Natural log of the survival function of the Burr type XII distribution.
    """
    with mp.extradps(5):
        c, d, scale = _validate_params(c, d, scale)
        x = mp.mpf(x)
        if x <= 0:
            return mp.zero
        return -d*mp.log1p((x/scale)**c)


def mean(c, d, scale):
    """
    Mean of the Burr type XII distribution.
    """
    with mp.extradps(5):
        c, d, scale = _validate_params(c, d, scale)
        if c*d <= 1:
            return mp.nan
        return d*mp.beta(d - 1/c, 1 + 1/c)*scale


def var(c, d, scale):
    """
    Variance of the Burr type XII distribution.
    """
    with mp.extradps(5):
        c, d, scale = _validate_params(c, d, scale)
        if c*d <= 2:
            return mp.nan
        mu1 = mean(c, d, 1)
        mu2 = d*mp.beta(d - 2/c, 1 + 2/c)
        return scale**2 * (-mu1**2 + mu2)


def median(c, d, scale):
    """
    Median of the Burr type XII distribution.
    """
    with mp.extradps(5):
        c, d, scale = _validate_params(c, d, scale)
        return scale * (2**(1/d) - 1)**(1/c)


def mode(c, d, scale):
    """
    Mode of the Burr type XII distribution.
    """
    with mp.extradps(5):
        c, d, scale = _validate_params(c, d, scale)
        if c <= 1:
            return mp.zero
        return scale*((c - 1)/(d*c + 1))**(1/c)
