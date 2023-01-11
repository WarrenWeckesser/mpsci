"""
Truncated standard normal distribution
--------------------------------------

The implementations here are somewhat naive.  To check the results,
run multiple times with increasing mpmath precision.
"""

from mpmath import mp
from . import normal
from ._common import _validate_p


__all__ = ['pdf', 'logpdf', 'cdf', 'sf', 'invcdf', 'invsf', 'mean', 'median',
           'var', 'skewness', 'kurtosis']


def _validate_params(a, b):
    if a >= b:
        raise ValueError("'a' must be less than 'b'")
    return mp.mpf(a), mp.mpf(b)


def _norm_delta_cdf(a, b):
    """
    Compute CDF(b) - CDF(a) for the standard normal distribution CDF.

    The function assumes a <= b.
    """
    with mp.extradps(5):
        if a == b:
            return mp.zero
        if a > 0:
            delta = mp.ncdf(-a) - mp.ncdf(-b)
        else:
            delta = mp.ncdf(b) - mp.ncdf(a)
        return delta


def pdf(x, a, b):
    """
    PDF of the truncated standard normal distribution.
    """
    with mp.extradps(5):
        a, b = _validate_params(a, b)
        x = mp.mpf(x)
        if x < a or x > b:
            return mp.zero
        delta = _norm_delta_cdf(a, b)
        return mp.npdf(x) / delta


def logpdf(x, a, b):
    """
    Natural logarithm of the PDF of the truncated standard normal distribution.
    """
    with mp.extradps(5):
        a, b = _validate_params(a, b)
        x = mp.mpf(x)
        if x < a or x > b:
            return mp.ninf
        delta = _norm_delta_cdf(a, b)
        return normal.logpdf(x) - mp.log(delta)


def cdf(x, a, b):
    """
    CDF of the truncated standard normal distribution.
    """
    with mp.extradps(5):
        a, b = _validate_params(a, b)
        x = mp.mpf(x)
        if x <= a:
            return mp.zero
        if x >= b:
            return mp.one
        return _norm_delta_cdf(a, x) / _norm_delta_cdf(a, b)


def sf(x, a, b):
    """
    Survival function of the truncated standard normal distribution.
    """
    with mp.extradps(5):
        a, b = _validate_params(a, b)
        x = mp.mpf(x)
        if x <= a:
            return mp.one
        if x >= b:
            return mp.zero
        return _norm_delta_cdf(x, b) / _norm_delta_cdf(a, b)


def invcdf(p, a, b):
    """
    Inverse of the CDF of the truncated standard normal distribution.

    This function is also known as the quantile function or the percent
    point function.
    """
    with mp.extradps(5):
        p = _validate_p(p)
        a, b = _validate_params(a, b)
        p2 = p * _norm_delta_cdf(a, b) + mp.ncdf(a)
        return normal.invcdf(p2)


def invsf(p, a, b):
    """
    Inverse of the survival function of the standard normal distribution.
    """
    with mp.extradps(5):
        p = _validate_p(p)
        a, b = _validate_params(a, b)
        p2 = -p * _norm_delta_cdf(a, b) + mp.ncdf(b)
        return normal.invcdf(p2)


def mean(a, b):
    """
    Mean of the truncated standard normal distribution.
    """
    with mp.extradps(5):
        a, b = _validate_params(a, b)
        return pdf(a, a, b) - pdf(b, a, b)


def median(a, b):
    """
    Median of the truncated standard normal distribution.
    """
    with mp.extradps(5):
        a, b = _validate_params(a, b)
        return normal.invcdf((normal.cdf(a) + normal.cdf(b))/2)


def var(a, b):
    """
    Variance of the truncated standard normal distribution.
    """
    with mp.extradps(5):
        a, b = _validate_params(a, b)
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
    with mp.extradps(5):
        a, b = _validate_params(a, b)
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
    with mp.extradps(5):
        a, b = _validate_params(a, b)
        m1, m2, m3 = _noncentral_moments(3, a, b)

        # Central moments (i.e. moments about the mean)
        mu2 = m2 - m1**2
        mu3 = m3 + m1 * (-3*m2 + 2*m1**2)

        # Skewness
        g1 = mu3 / mp.power(mu2, 1.5)

        return g1


def kurtosis(a, b):
    """
    Excess kurtosis of the truncated standard normal distribution.
    """
    with mp.extradps(5):
        a, b = _validate_params(a, b)
        m1, m2, m3, m4 = _noncentral_moments(4, a, b)

        # Central moments (i.e. moments about the mean)
        mu2 = m2 - m1**2
        mu4 = m4 + m1*(-4*m3 + 3*m1*(2*m2 - m1**2))

        # Excess kurtosis
        g2 = mu4 / mu2**2 - 3

        return g2
