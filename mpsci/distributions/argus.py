"""
ARGUS distribution
------------------

The parameters, chi and c, follow the conventions used in the
wikipedia article https://en.wikipedia.org/wiki/ARGUS_distribution.

Note that c is a scale parameter.
"""

import mpmath


__all__ = ['pdf', 'logpdf', 'cdf', 'sf', 'mean', 'var', 'mode']


def _psi(chi):
    return mpmath.ncdf(chi) - chi*mpmath.npdf(chi) - mpmath.mpf('0.5')


def pdf(x, chi, c):
    """
    PDF of the ARGUS probability distribution.
    """
    if c <= 0:
        raise ValueError('c must be positive')
    if chi <= 0:
        raise ValueError('chi must be positive')

    if x < 0 or x > c:
        return mpmath.mp.zero

    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        chi = mpmath.mpf(chi)
        c = mpmath.mpf(c)
        return mpmath.exp(logpdf(x, chi, c))


def logpdf(x, chi, c):
    """
    Logarithm of the PDF of the ARGUS probability distribution.
    """
    if c <= 0:
        raise ValueError('c must be positive')
    if chi <= 0:
        raise ValueError('chi must be positive')

    if x < 0 or x > c:
        return mpmath.mp.ninf

    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        chi = mpmath.mpf(chi)
        c = mpmath.mpf(c)
        z = x/c
        t1 = (3*mpmath.log(chi)
              - mpmath.log(2*mpmath.pi)/2
              - mpmath.log(_psi(chi)))
        t2 = -mpmath.log(c) + mpmath.log(z)
        t3 = mpmath.log1p(-z**2)/2
        t4 = -chi**2/2*(1 - z**2)
        return t1 + t2 + t3 + t4


def cdf(x, chi, c):
    """
    CDF of the ARGUS probability distribution.
    """
    if c <= 0:
        raise ValueError('c must be positive')
    if chi <= 0:
        raise ValueError('chi must be positive')

    if x < 0:
        return mpmath.mp.zero
    if x > c:
        return mpmath.mp.one

    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        chi = mpmath.mpf(chi)
        c = mpmath.mpf(c)
        z = x/c
        return (mpmath.mp.one -
                _psi(chi*mpmath.sqrt(1 - z**2)) / _psi(chi))


def sf(x, chi, c):
    """
    Survival function of the ARGUS probability distribution.
    """
    if c <= 0:
        raise ValueError('c must be positive')
    if chi <= 0:
        raise ValueError('chi must be positive')

    if x < 0:
        return mpmath.mp.one
    if x > c:
        return mpmath.mp.zero

    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        chi = mpmath.mpf(chi)
        c = mpmath.mpf(c)
        z = x/c
        return _psi(chi*mpmath.sqrt(1 - z**2)) / _psi(chi)


def mean(chi, c):
    """
    Mean of the ARGUS distribution.
    """
    if c <= 0:
        raise ValueError('c must be positive')
    if chi <= 0:
        raise ValueError('chi must be positive')

    with mpmath.extradps(5):
        chi = mpmath.mpf(chi)
        c = mpmath.mpf(c)
        chi2o4 = chi**2/4
        p1 = c*mpmath.sqrt(mpmath.pi/8)
        p2 = chi*mpmath.exp(-chi2o4)
        p3 = mpmath.besseli(1, chi2o4)
        return p1 * p2 * p3 / _psi(chi)


def var(chi, c):
    """
    Variance of the ARGUS distribution.
    """
    if c <= 0:
        raise ValueError('c must be positive')
    if chi <= 0:
        raise ValueError('chi must be positive')

    with mpmath.extradps(5):
        chi = mpmath.mpf(chi)
        c = mpmath.mpf(c)
        mu = mean(chi, c)
        t1 = c**2 * (mpmath.mp.one - 3/chi**2 + chi*mpmath.npdf(chi)/_psi(chi))
        return t1 - mu**2


def mode(chi, c):
    """
    Mode of the ARGUS distribution.
    """
    if c <= 0:
        raise ValueError('c must be positive')
    if chi <= 0:
        raise ValueError('chi must be positive')

    with mpmath.extradps(5):
        chi = mpmath.mpf(chi)
        c = mpmath.mpf(c)
        chi2 = chi**2
        p1 = c/mpmath.sqrt(2)/chi
        p2 = mpmath.sqrt(chi2 - 2 + mpmath.sqrt(chi2**2 + 4))
        return p1 * p2
