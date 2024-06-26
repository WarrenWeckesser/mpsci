"""
Relativistic Breit-Wigner distribution
--------------------------------------

* `rho` is the shape parameter.
* `scale` is (not surprisingly) the scale parameter.  This is a true scale
  parameter, in the sense of https://en.wikipedia.org/wiki/Scale_parameter.

In the wikipedia article
https://en.wikipedia.org/wiki/Relativistic_Breit%E2%80%93Wigner_distribution,
the parameters are M and Γ.  The relations between those parameters and
the parameters use here are:

* from (M, Γ) to mpsci's (rho, scale)::

      rho = M/Γ
      scale = Γ

* from mpsci's (rho, scale) to (M, Γ)::

      M = rho * scale
      Γ = scale


See https://github.com/scipy/scipy/issues/6414 for the details.

"""

from mpmath import mp
from ..fun import pow1pm1
from ._common import _validate_p, _validate_moment_n,  _find_bracket


__all__ = ['pdf', 'logpdf', 'cdf', 'invcdf', 'sf', 'invsf',
           'support', 'mean', 'var', 'mode', 'noncentral_moment']


def _validate_rho_scale(rho, scale):
    if rho <= 0:
        raise ValueError('rho must be positive')
    if scale <= 0:
        raise ValueError('scale must be positive')
    return mp.mpf(rho), mp.mpf(scale)


def _k(rho):
    # FIXME: `_k` is a horrible name!
    with mp.extradps(5):
        rho = mp.mpf(rho)
        rho2 = rho**2
        s = mp.sqrt(rho2 + 1)
        return 2*mp.sqrt(2)*rho2*s / mp.sqrt(rho2 + rho*s) / mp.pi


def pdf(x, rho, scale):
    """
    Probability density function of the relativistic Breit-Wigner distribution.
    """
    with mp.extradps(5):
        x = mp.mpf(x)
        rho, scale = _validate_rho_scale(rho, scale)
        if x < 0:
            return mp.zero
        k = _k(rho)
        rho2 = rho**2
        z = x / scale
        return k / ((z**2 - rho2)**2 + rho2) / scale


def logpdf(x, rho, scale):
    """
    Logarithm of the PDF of the relativistic Breit-Wigner distribution.
    """
    with mp.extradps(5):
        x = mp.mpf(x)
        rho, scale = _validate_rho_scale(rho, scale)
        if x < 0:
            return mp.ninf
        logk = mp.log(_k(rho))
        z = x / scale
        return (logk - 2*mp.log(rho) - mp.log(scale)
                - mp.log1p(((z + rho)*(z - rho)/rho)**2))


def cdf(x, rho, scale):
    """
    Cumulative distribution function of the relativistic Breit-Wigner distr.
    """
    with mp.extradps(5):
        x = mp.mpf(x)
        rho, scale = _validate_rho_scale(rho, scale)
        if x < 0:
            return mp.zero
        if mp.isinf(x):
            return mp.one
        z = x/scale
        k = _k(rho)
        alpha = mp.sqrt(-rho*(rho + 1j))
        return k*(mp.atan(z/alpha)/alpha).imag/rho


def invcdf(p, rho, scale):
    """
    Inverse of the CDF of the relativistic Breit-Wigner distribution.

    The implementation uses a numerical root finder, so it may be slow, and
    it may fail to converge for some inputs.
    """
    with mp.extradps(5):
        p = _validate_p(p)
        rho, scale = _validate_rho_scale(rho, scale)
        x0, x1 = _find_bracket(lambda x: cdf(x, rho, scale), p, 0, mp.inf)
        root = mp.findroot(lambda t: cdf(t, rho, scale) - p, x0=(x0, x1))
        return root


def sf(x, rho, scale):
    """
    Survival function of the relativistic Breit-Wigner distribution.
    """
    # Double the precision and return 1 - CDF.  Expensive, but easy :)
    with mp.extradps(mp.dps):
        return 1 - cdf(x, rho, scale)


def invsf(p, rho, scale):
    """
    Inverse of the survival function of the relativistic Breit-Wigner distr.

    The implementation uses a numerical root finder, so it may be slow, and
    it may fail to converge for some inputs.
    """
    with mp.extradps(5):
        p = _validate_p(p)
        rho, scale = _validate_rho_scale(rho, scale)
        x0, x1 = _find_bracket(lambda x: sf(x, rho, scale), p, 0, mp.inf)
        root = mp.findroot(lambda t: sf(t, rho, scale) - p, x0=(x0, x1))
        return root


def support(rho, scale):
    """
    Support of the relativistic Breit-Wigner distribution.
    """
    with mp.extradps(5):
        rho, scale = _validate_rho_scale(rho, scale)
        return (mp.zero, mp.inf)


def mean(rho, scale):
    """
    Mean of the relativistic Breit-Wigner distribution.
    """
    with mp.extradps(5):
        rho, scale = _validate_rho_scale(rho, scale)
        k = _k(rho)
        return scale * k / (2*rho) * (mp.pi/2 + mp.atan(rho))


def var(rho, scale):
    """
    Variance of the relativistic Breit-Wigner distribution.
    """
    with mp.extradps(5):
        rho, scale = _validate_rho_scale(rho, scale)
        return noncentral_moment(2, rho, scale) - mean(rho, scale)**2


def mode(rho, scale):
    """
    Mode of the relativistic Breit-Wigner distribution.
    """
    rho, scale = _validate_rho_scale(rho, scale)
    return rho*scale


def noncentral_moment(n, rho, scale):
    """
    n-th noncentral moment of the relativistic Breit-Wigner distribution.

    n must be a nonnegative integer.  For n > 2, the noncentral moment
    diverges, so `inf` is returned.
    """
    n = _validate_moment_n(n)
    rho, scale = _validate_rho_scale(rho, scale)
    if n == 0:
        return mp.one
    if n == 1:
        return mean(rho, scale)
    if n == 2:
        with mp.extradps(5):
            c = mp.pi/(2*mp.sqrt(2))
            k = _k(rho)
            half = mp.one/2
            return scale**2*c*k/(rho*mp.sqrt(pow1pm1(rho**-2, half)))
    return mp.inf
