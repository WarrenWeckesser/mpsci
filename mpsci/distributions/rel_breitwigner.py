"""
Relativistic Breit-Wigner distribution
--------------------------------------

* `rho` is the shape parameter.
* `scale` is (not surprisingly) the scale parameter.  This is a true scale
  parameter, in the sense of https://en.wikipedia.org/wiki/Scale_parameter.

In the wikipedia article
https://en.wikipedia.org/wiki/Relativistic_Breit%E2%80%93Wigner_distribution,
the parameters are M and Γ.  The relations between those parmeters and
the parameters use here are:

* from (M, Γ) to mpsci's (rho, scale)::

      rho = M/Γ
      scale = Γ

* from mpsci's (rho, scale) to (M, Γ)::

      M = rho * scale
      Γ = scale


See https://github.com/scipy/scipy/issues/6414 for the details.

"""

import mpmath
from mpmath import mp

from ._common import _validate_p,  _find_bracket


__all__ = ['pdf', 'cdf', 'invcdf', 'mean', 'mode']


def _validate_rho_scale(rho, scale):
    if rho <= 0:
        raise ValueError('rho must be positive')
    if scale <= 0:
        raise ValueError('scale must be positive')
    return mp.mpf(rho), mp.mpf(scale)


def _k(rho):
    with mpmath.extradps(5):
        rho = mp.mpf(rho)
        rho2 = rho**2
        s = mp.sqrt(rho2 + 1)
        return 2*mp.sqrt(2)*rho2*s / mp.sqrt(rho2 + rho*s) / mp.pi


def pdf(x, rho, scale):
    """
    Probability density function of the relativistic Breit-Wigner distribution.
    """
    with mpmath.extradps(5):
        x = mp.mpf(x)
        rho, scale = _validate_rho_scale(rho, scale)
        if x < 0:
            return mp.zero
        k = _k(rho)
        rho2 = rho**2
        z = x / scale
        return k / ((z**2 - rho2)**2 + rho2) / scale


def cdf(x, rho, scale):
    """
    Cumulative distribution function of the relativistic Breit-Wigner distr.
    """
    with mpmath.extradps(5):
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
    with mpmath.extradps(5):
        p = _validate_p(p)
        rho, scale = _validate_rho_scale(rho, scale)
        x0, x1 = _find_bracket(lambda x: cdf(x, rho, scale), p, 0, mp.inf)
        root = mp.findroot(lambda t: cdf(t, rho, scale) - p, x0=(x0, x1))
        return root


def mean(rho, scale):
    """
    Mean of the relativistic Breit-Wigner distribution.
    """
    with mpmath.extradps(5):
        rho, scale = _validate_rho_scale(rho, scale)
        k = _k(rho)
        return scale * k / (2*rho) * (mp.pi/2 + mp.atan(rho))


def mode(rho, scale):
    """
    Mode of the relativistic Breit-Wigner distribution.
    """
    rho, scale = _validate_rho_scale(rho, scale)
    return rho*scale
