"""
ARGUS distribution
------------------

The parameters, chi and c, follow the conventions used in the
wikipedia article https://en.wikipedia.org/wiki/ARGUS_distribution.

Note that c is a scale parameter.
"""

from mpmath import mp


__all__ = ['pdf', 'logpdf', 'cdf', 'sf', 'support', 'mean', 'var', 'mode']


def _validate_params(chi, c):
    if chi <= 0:
        raise ValueError('chi must be positive')
    if c <= 0:
        raise ValueError('c must be positive')
    return mp.mpf(chi), mp.mpf(c)


def _psi(chi):
    return mp.ncdf(chi) - chi*mp.npdf(chi) - mp.mpf('0.5')


def pdf(x, chi, c):
    """
    PDF of the ARGUS probability distribution.
    """
    with mp.extradps(5):
        chi, c = _validate_params(chi, c)
        x = mp.mpf(x)
        if x < 0 or x > c:
            return mp.zero
        return mp.exp(logpdf(x, chi, c))


def logpdf(x, chi, c):
    """
    Logarithm of the PDF of the ARGUS probability distribution.
    """
    with mp.extradps(5):
        chi, c = _validate_params(chi, c)
        x = mp.mpf(x)
        if x < 0 or x > c:
            return mp.ninf
        z = x/c
        t1 = 3*mp.log(chi) - mp.log(2*mp.pi)/2 - mp.log(_psi(chi))
        t2 = -mp.log(c) + mp.log(z)
        t3 = mp.log1p(-z**2)/2
        t4 = -chi**2/2*(1 - z**2)
        return t1 + t2 + t3 + t4


def cdf(x, chi, c):
    """
    CDF of the ARGUS probability distribution.
    """
    with mp.extradps(5):
        chi, c = _validate_params(chi, c)
        x = mp.mpf(x)
        if x < 0:
            return mp.zero
        if x > c:
            return mp.one
        z = x/c
        return mp.one - _psi(chi*mp.sqrt(1 - z**2)) / _psi(chi)


def sf(x, chi, c):
    """
    Survival function of the ARGUS probability distribution.
    """
    with mp.extradps(5):
        chi, c = _validate_params(chi, c)
        x = mp.mpf(x)
        if x < 0:
            return mp.one
        if x > c:
            return mp.zero
        z = x/c
        return _psi(chi*mp.sqrt(1 - z**2)) / _psi(chi)


def support(chi, c):
    """
    Support of the ARGUS probability distribution.
    """
    with mp.extradps(5):
        chi, c = _validate_params(chi, c)
        return (mp.zero, c)


def mean(chi, c):
    """
    Mean of the ARGUS distribution.
    """
    with mp.extradps(5):
        chi, c = _validate_params(chi, c)
        chi2o4 = chi**2/4
        p1 = c*mp.sqrt(mp.pi/8)
        p2 = chi*mp.exp(-chi2o4)
        p3 = mp.besseli(1, chi2o4)
        return p1 * p2 * p3 / _psi(chi)


def var(chi, c):
    """
    Variance of the ARGUS distribution.
    """
    with mp.extradps(5):
        chi, c = _validate_params(chi, c)
        mu = mean(chi, c)
        t1 = c**2 * (mp.one - 3/chi**2 + chi*mp.npdf(chi)/_psi(chi))
        return t1 - mu**2


def mode(chi, c):
    """
    Mode of the ARGUS distribution.
    """
    with mp.extradps(5):
        chi, c = _validate_params(chi, c)
        chi2 = chi**2
        p1 = c/mp.sqrt(2)/chi
        p2 = mp.sqrt(chi2 - 2 + mp.sqrt(chi2**2 + 4))
        return p1 * p2
