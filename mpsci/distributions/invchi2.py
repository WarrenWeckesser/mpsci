"""
Inverse chi-square distribution
-------------------------------

See https://en.wikipedia.org/wiki/Inverse-chi-squared_distribution

"""

import mpmath


__all__ = ['pdf', 'logpdf', 'cdf', 'sf', 'mean', 'mode', 'variance']


def _validate_nu(nu):
    if nu <= 0:
        raise ValueError('nu must be positive')


def pdf(x, nu):
    """
    PDF for the inverse chi-square distribution.
    """
    _validate_nu(nu)
    if x <= 0:
        return mpmath.mp.zero
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        nu = mpmath.mpf(nu)
        hnu = nu/2
        p = (mpmath.power(2, -hnu) * x**(-hnu - 1) * mpmath.exp(-1/(2*x))
             / mpmath.gamma(hnu))
        return p


def logpdf(x, nu):
    """
    Logarithm of the PDF for the inverse chi-square distribution.
    """
    _validate_nu(nu)
    if x <= 0:
        return mpmath.ninf
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        nu = mpmath.mpf(nu)
        hnu = nu/2
        logp = (-hnu*mpmath.log(2) + (-hnu - 1)*mpmath.log(x) - 1/(2*x)
                - mpmath.loggamma(hnu))
        return logp


def cdf(x, nu):
    """
    CDF for the inverse chi-square distribution.
    """
    _validate_nu(nu)
    if x <= 0:
        return mpmath.mp.zero
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        nu = mpmath.mpf(nu)
        c = mpmath.gammainc(nu/2, a=1/(2*x), b=mpmath.inf, regularized=True)
    return c


def sf(x, nu):
    """
    Survival function for the inverse chi-square distribution.
    """
    _validate_nu(nu)
    if x <= 0:
        return mpmath.mp.one
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        nu = mpmath.mpf(nu)
        s = mpmath.gammainc(nu/2, a=0, b=1/(2*x), regularized=True)
    return s


def mean(nu):
    """
    Mean of the inverse chi-square distribution.
    """
    _validate_nu(nu)
    with mpmath.extradps(5):
        nu = mpmath.mpf(nu)
        return mpmath.mp.one / (nu - 2) if nu > 2 else mpmath.nan


def mode(nu):
    """
    Mode of the inverse chi-square distribution.

    The mode is max(k - 2, 0).
    """
    _validate_nu(nu)
    with mpmath.extradps(5):
        nu = mpmath.mpf(nu)
        return 1 / (nu + 2)


def variance(nu):
    """
    Variance of the inverse chi-square distribution.
    """
    _validate_nu(nu)
    with mpmath.extradps(5):
        nu = mpmath.mpf(nu)
        return 2/(nu - 2)**2 / (nu - 4) if nu > 4 else mpmath.nan
