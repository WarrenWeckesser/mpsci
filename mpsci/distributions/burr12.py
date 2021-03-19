"""
Burr type XII probability distribution
--------------------------------------
"""

import mpmath


__all__ = ['pdf', 'logpdf', 'cdf', 'sf', 'logsf', 'mean', 'var',
           'median', 'mode']


def _validate_params(c, d, scale):
    if c <= 0:
        raise ValueError('c must be greater than 0.')
    if d <= 0:
        raise ValueError('d must be greater than 0.')
    if scale <= 0:
        raise ValueError('scale must be greater than 0.')


def pdf(x, c, d, scale):
    """

    Unlike scipy, a location parameter is not included.
    """
    _validate_params(c, d, scale)
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        c = mpmath.mpf(c)
        d = mpmath.mpf(d)
        scale = mpmath.mpf(scale)
        z = x/scale
        return c*d*z**(c - 1)/scale / (1 + z**c)**(d+1)


def logpdf(x, c, d, scale):
    """
    Log of the PDF of the Burr type XII distribution.
    """
    _validate_params(c, d, scale)
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        c = mpmath.mpf(c)
        d = mpmath.mpf(d)
        scale = mpmath.mpf(scale)
        return (mpmath.log(c) + mpmath.log(d) + (c - 1)*mpmath.log(x)
                - c*mpmath.log(scale) - (d + 1)*mpmath.log1p((x / scale)**c))


def cdf(x, c, d, scale):
    """
    Burr type XII distribution cumulative distribution function.

    Unlike scipy, a location parameter is not included.
    """
    _validate_params(c, d, scale)
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        c = mpmath.mpf(c)
        d = mpmath.mpf(d)
        scale = mpmath.mpf(scale)
        # TO DO: See if the use of logsf (as in scipy) is worthwhile.
        return 1 - sf(x, c, d, scale)


def sf(x, c, d, scale):
    """
    Survival function of the Burr type XII distribution.
    """
    _validate_params(c, d, scale)
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        c = mpmath.mpf(c)
        d = mpmath.mpf(d)
        scale = mpmath.mpf(scale)
        if x < 0:
            return mpmath.mp.one
        return (1 + (x/scale)**c)**(-d)


def logsf(x, c, d, scale):
    """
    Natural log of the survival function of the Burr type XII distribution.
    """
    _validate_params(c, d, scale)
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        c = mpmath.mpf(c)
        d = mpmath.mpf(d)
        scale = mpmath.mpf(scale)
        if x < 0:
            return mpmath.ninf
        return -d*mpmath.log1p((x/scale)**c)


def mean(c, d, scale):
    """
    Mean of the Burr type XII distribution.
    """
    _validate_params(c, d, scale)
    with mpmath.extradps(5):
        c = mpmath.mpf(c)
        d = mpmath.mpf(d)
        scale = mpmath.mpf(scale)
        return d*mpmath.beta(d - 1/c, 1 + 1/c)*scale


def var(c, d, scale):
    """
    Variance of the Burr type XII distribution.
    """
    _validate_params(c, d, scale)
    with mpmath.extradps(5):
        c = mpmath.mpf(c)
        d = mpmath.mpf(d)
        scale = mpmath.mpf(scale)
        mu1 = mean(c, d, 1)
        mu2 = d*mpmath.beta(d - 2/c, 1 + 2/c)
        return scale**2 * (-mu1**2 + mu2)


def median(c, d, scale):
    """
    Median of the Burr type XII distribution.
    """
    _validate_params(c, d, scale)
    with mpmath.extradps(5):
        c = mpmath.mpf(c)
        d = mpmath.mpf(d)
        scale = mpmath.mpf(scale)
        return scale * (2**(1/d) - 1)**(1/c)


def mode(c, d, scale):
    """
    Mode of the Burr type XII distribution.
    """
    _validate_params(c, d, scale)
    with mpmath.extradps(5):
        c = mpmath.mpf(c)
        d = mpmath.mpf(d)
        scale = mpmath.mpf(scale)
        if c <= 1:
            return mpmath.mp.zero
        return scale*((c - 1)/(d*c + 1))**(1/c)
