"""
Truncated standard normal distribution
--------------------------------------

The implementations here are somewhat naive.  To check the results,
run multiple times with increasing mpmath precision.
"""

import mpmath
from . import normal


__all__ = ['pdf', 'logpdf', 'cdf', 'sf', 'invcdf', 'invsf', 'mean', 'var',
           'median']


def _validate_params(a, b):
    if a >= b:
        raise ValueError("'a' must be less than 'b'")


def _norm_delta_cdf(a, b):
    """
    Compute CDF(b) - CDF(a) for the standard normal distribution CDF.

    The function assumes a <= b.
    """
    with mpmath.extradps(5):
        if a == b:
            return mpmath.mp.zero
        if a > 0:
            delta = mpmath.ncdf(-a) - mpmath.ncdf(-b)
        else:
            delta = mpmath.ncdf(b) - mpmath.ncdf(a)
        return delta


def pdf(x, a, b):
    """
    PDF of the truncated standard normal distribution.
    """
    _validate_params(a, b)
    if x < a or x > b:
        return mpmath.mp.zero
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        a = mpmath.mpf(a)
        b = mpmath.mpf(b)
        delta = _norm_delta_cdf(a, b)
        return mpmath.npdf(x) / delta


def logpdf(x, a, b):
    """
    Natural logarithm of the PDF of the truncated standard normal distribution.
    """
    _validate_params(a, b)
    if x < a or x > b:
        return -mpmath.inf
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        a = mpmath.mpf(a)
        b = mpmath.mpf(b)
        delta = _norm_delta_cdf(a, b)
        return normal.logpdf(x) - mpmath.log(delta)


def cdf(x, a, b):
    """
    CDF of the truncated standard normal distribution.
    """
    _validate_params(a, b)
    if x <= a:
        return mpmath.mp.zero
    if x >= b:
        return mpmath.mp.one
    with mpmath.extradps(5):
        return _norm_delta_cdf(a, x) / _norm_delta_cdf(a, b)


def sf(x, a, b):
    """
    Survival function of the truncated standard normal distribution.
    """
    _validate_params(a, b)
    if x <= a:
        return mpmath.mp.one
    if x >= b:
        return mpmath.mp.zero
    with mpmath.extradps(5):
        return _norm_delta_cdf(x, b) / _norm_delta_cdf(a, b)


def invcdf(p, a, b):
    """
    Inverse of the CDF of the truncated standard normal distribution.

    This function is also known as the quantile function or the percent
    point function.
    """
    _validate_params(a, b)
    if p < 0 or p > 1:
        return mpmath.nan

    with mpmath.extradps(5):
        p = mpmath.mpf(p)
        a = mpmath.mpf(a)
        b = mpmath.mpf(b)

        p2 = p * _norm_delta_cdf(a, b) + mpmath.ncdf(a)
        x = normal.invcdf(p2)

    return x


def invsf(p, a, b):
    """
    Inverse of the survival function of the standard normal distribution.
    """
    _validate_params(a, b)
    if p < 0 or p > 1:
        return mpmath.nan

    with mpmath.extradps(5):
        p = mpmath.mpf(p)
        a = mpmath.mpf(a)
        b = mpmath.mpf(b)

        p2 = -p * _norm_delta_cdf(a, b) + mpmath.ncdf(b)
        x = normal.invcdf(p2)

    return x


def mean(a, b):
    """
    Mean of the truncated standard normal distribution.
    """
    _validate_params(a, b)
    with mpmath.extradps(5):
        return pdf(a, a, b) - pdf(b, a, b)


def median(a, b):
    """
    Median of the truncated standard normal distribution.
    """
    _validate_params(a, b)
    with mpmath.extradps(5):
        return normal.invcdf((normal.cdf(a) + normal.cdf(b))/2)


def var(a, b):
    """
    Variance of the truncated standard normal distribution.
    """
    _validate_params(a, b)
    with mpmath.extradps(5):
        a = mpmath.mpf(a)
        b = mpmath.mpf(b)
        pa = pdf(a, a, b)
        pb = pdf(b, a, b)
        # Avoid the possibility of inf*0:
        ta = 0 if pa == 0 else a*pa
        tb = 0 if pb == 0 else b*pb
        return 1 + ta - tb - (pa - pb)**2


def _noncentral_moments(n, a, b):
    """
    Compute the noncentral moments m1, m2, ... mn.

    These are the moments about 0.

    Returns a list of length n.  n must be a positive integer.
    """
    _validate_params(a, b)
    with mpmath.extradps(5):
        a = mpmath.mpf(a)
        b = mpmath.mpf(b)

        pa = pdf(a, a, b)
        pb = pdf(b, a, b)

        # Avoid the possibility of inf*0:
        if pa == 0:
            a = 0
        if pb == 0:
            b = 0
        apa = a*pa
        bpb = b*pb

        result = [pa - pb]
        if n >= 2:
            result.append(1 + apa - bpb)
        for k in range(3, n+1):
            apa = a*apa
            bpb = b*bpb
            result.append((k-1)*result[k-3] + apa - bpb)

        return result


def skewness(a, b):
    """
    Skewness of the truncated standard normal distribution.
    """
    _validate_params(a, b)
    with mpmath.extradps(5):
        m1, m2, m3 = _noncentral_moments(3, a, b)

        # Central moments (i.e. moments about the mean)
        mu2 = m2 - m1**2
        mu3 = m3 + m1 * (-3*m2 + 2*m1**2)

        # Skewness
        g1 = mu3 / mpmath.power(mu2, 1.5)

        return g1


def kurtosis(a, b):
    """
    Excess kurtosis of the truncated standard normal distribution.
    """
    _validate_params(a, b)
    with mpmath.extradps(5):
        m1, m2, m3, m4 = _noncentral_moments(4, a, b)

        # Central moments (i.e. moments about the mean)
        mu2 = m2 - m1**2
        mu4 = m4 + m1*(-4*m3 + 3*m1*(2*m2 - m1**2))

        # Excess kurtosis
        g2 = mu4 / mu2**2 - 3

        return g2
