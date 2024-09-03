"""
ARGUS distribution
------------------

The parameters, chi and c, follow the conventions used in the
wikipedia article https://en.wikipedia.org/wiki/ARGUS_distribution.

Note that the parameter c in the wikipedia article is called ``scale``
here.
"""

from mpmath import mp


__all__ = ['pdf', 'logpdf', 'cdf', 'sf', 'support', 'mean', 'var', 'mode']


def _validate_params(chi, scale):
    if chi < 0:
        raise ValueError('chi must be nonnegative')
    if scale <= 0:
        raise ValueError('scale must be positive')
    return mp.mpf(chi), mp.mpf(scale)


def _powm1(x, b):
    return mp.expm1(b*mp.log(x))


def _psi(chi):
    return mp.ncdf(chi) - chi*mp.npdf(chi) - mp.mpf('0.5')


def pdf(x, chi, scale):
    """
    PDF of the ARGUS probability distribution.
    """
    with mp.extradps(5):
        chi, scale = _validate_params(chi, scale)
        x = mp.mpf(x)
        if x < 0 or x > scale:
            return mp.zero
        return mp.exp(logpdf(x, chi, scale))


def logpdf(x, chi, scale):
    """
    Logarithm of the PDF of the ARGUS probability distribution.
    """
    with mp.extradps(5):
        chi, scale = _validate_params(chi, scale)
        x = mp.mpf(x)
        if x < 0 or x > scale:
            return mp.ninf
        z = x/scale
        if chi == 0:
            return (mp.log(3) + mp.log(z) + 0.5*(mp.log1p(-z) + mp.log1p(z))
                    - mp.log(scale))
        else:
            t1 = 3*mp.log(chi) - mp.log(2*mp.pi)/2 - mp.log(_psi(chi))
            t2 = -mp.log(scale) + mp.log(z)
            t3 = mp.log1p(-z**2)/2
            t4 = -chi**2/2*(1 - z)*(1 + z)
            return t1 + t2 + t3 + t4


def cdf(x, chi, scale):
    """
    CDF of the ARGUS probability distribution.
    """
    with mp.extradps(5):
        chi, scale = _validate_params(chi, scale)
        x = mp.mpf(x)
        if x < 0:
            return mp.zero
        if x > scale:
            return mp.one
        z = x/scale
        if chi == 0:
            # 1 - (1 - z**2)**1.5
            return -_powm1((1 + z)*(1 - z), 1.5)
        else:
            return mp.one - _psi(chi*mp.sqrt((1 - z)*(1 + z))) / _psi(chi)


def sf(x, chi, scale):
    """
    Survival function of the ARGUS probability distribution.
    """
    with mp.extradps(5):
        chi, scale = _validate_params(chi, scale)
        x = mp.mpf(x)
        if x < 0:
            return mp.one
        if x > scale:
            return mp.zero
        z = x/scale
        if chi == 0:
            return mp.power((1 + z)*(1 - z), 1.5)
        else:
            return _psi(chi*mp.sqrt((1 - z)*(1 + z))) / _psi(chi)


def support(chi, scale):
    """
    Support of the ARGUS probability distribution.
    """
    with mp.extradps(5):
        chi, scale = _validate_params(chi, scale)
        return (mp.zero, scale)


def mean(chi, scale):
    """
    Mean of the ARGUS distribution.
    """
    with mp.extradps(5):
        chi, scale = _validate_params(chi, scale)
        if chi == 0:
            return scale*3*mp.pi/16
        else:
            chi2o4 = chi**2/4
            p1 = scale*mp.sqrt(mp.pi/8)
            p2 = chi*mp.exp(-chi2o4)
            p3 = mp.besseli(1, chi2o4)
            return p1 * p2 * p3 / _psi(chi)


def var(chi, scale):
    """
    Variance of the ARGUS distribution.
    """
    with mp.extradps(5):
        chi, scale = _validate_params(chi, scale)
        mu = mean(chi, scale)
        if chi == 0:
            # TO DO: Derive this value.
            return scale**2*(mp.mpf('2/5') - 9*mp.pi**2/256)
        else:
            t1 = scale**2 * (mp.one - 3/chi**2 + chi*mp.npdf(chi)/_psi(chi))
            return t1 - mu**2


def mode(chi, scale):
    """
    Mode of the ARGUS distribution.
    """
    with mp.extradps(5):
        chi, scale = _validate_params(chi, scale)
        if chi == 0:
            return scale/mp.sqrt(2)
        else:
            chi2 = chi**2
            p1 = scale/mp.sqrt(2)/chi
            p2 = mp.sqrt(chi2 - 2 + mp.sqrt(chi2**2 + 4))
            return p1 * p2
