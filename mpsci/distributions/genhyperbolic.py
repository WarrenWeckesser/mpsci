"""
Generalized hyperbolic distribution
-----------------------------------

This implementation uses the same parametrization as
`scipy.stats.genhyperbolic` in SciPy.  The parametrization used here is
a true location/scale family. This is not the same as, for example, the
parametrization used in the wikipedia article
https://en.wikipedia.org/wiki/Generalised_hyperbolic_distribution.


A disadvantage of this parametrization is that, because the
parameter δ has been "scaled away", the variance gamma distribution
cannot be obtained as a special case of this distribution.  In the
parametrization used in the wikipedia article, the variance gamma
distribution is obtained by setting δ = 0.
"""

from mpmath import mp
from ._common import _validate_loc_scale


__all__ = ['logpdf', 'pdf', 'cdf', 'sf', 'support', 'mean', 'var']


def _validate_params(p, a, b, loc, scale):
    loc, scale = _validate_loc_scale(loc, scale)
    if a <= 0:
        raise ValueError('a must be greater than 0.')
    if abs(b) > a or (abs(b) == a and p >= 0):
        raise ValueError('must have abs(b) < a if p >= 0, '
                         'abs(b) <= a if p < 0')
    return mp.mpf(p), mp.mpf(a), mp.mpf(b), mp.mpf(loc), mp.mpf(scale)


@mp.extradps(5)
def logpdf(x, p, a, b, loc=0, scale=1):
    """
    Logarithm of the PDF of the generalized hyperbolic distribution.
    """
    p, a, b, loc, scale = _validate_params(p, a, b, loc, scale)
    x = mp.mpf(x)
    z = (x - loc)/scale
    z2 = z**2
    pmh = p - 0.5
    terms = [(p/2)*(mp.log(a + b) + mp.log(a - b)),
             b*z,
             mp.log(mp.besselk(pmh, a*mp.sqrt(1 + z2))),
             -0.5*mp.log(2*mp.pi),
             -pmh*mp.log(a),
             -mp.log(mp.besselk(p, mp.sqrt((a + b)*(a - b)))),
             0.5*pmh*mp.log1p(z2),
             -mp.log(scale)]
    return mp.fsum(terms)


@mp.extradps(5)
def pdf(x, p, a, b, loc=0, scale=1):
    """
    Probability density function of the generalized hyperbolic distribution.
    """
    return mp.exp(logpdf(x, p, a, b, loc, scale))


def _integrate_pdf(x0, x1, p, a, b, loc, scale):
    m = mean(p, a, b, loc, scale)
    if x0 < m < x1:
        q1 = mp.quad(lambda t: pdf(t, p, a, b, loc, scale), [x0, m])
        q2 = mp.quad(lambda t: pdf(t, p, a, b, loc, scale), [m, x1])
        return q1 + q2
    else:
        return mp.quad(lambda t: pdf(t, p, a, b, loc, scale), [x0, x1])


@mp.extradps(5)
def cdf(x, p, a, b, loc=0, scale=1):
    """
    Cumulative distribution function of the generalized hyperbolic distr.

    The result is computed by numerically integrating the PDF, so it is
    relatively slow.
    """
    p, a, b, loc, scale = _validate_params(p, a, b, loc, scale)
    x = mp.mpf(x)
    return _integrate_pdf(-mp.inf, x, p, a, b, loc, scale)


@mp.extradps(5)
def sf(x, p, a, b, loc=0, scale=1):
    """
    Survival function of the generalized hyperbolic distr.

    The result is computed by numerically integrating the PDF, so it is
    relatively slow.
    """
    p, a, b, loc, scale = _validate_params(p, a, b, loc, scale)
    x = mp.mpf(x)
    return _integrate_pdf(x, mp.inf, p, a, b, loc, scale)


@mp.extradps(5)
def support(p, a, b, loc=0, scale=1):
    """
    Support of the generalized hyperbolic distribution.
    """
    p, a, b, loc, scale = _validate_params(p, a, b, loc, scale)
    return (mp.ninf, mp.inf)


@mp.extradps(5)
def mean(p, a, b, loc=0, scale=1):
    """
    Mean of the generalized hyperbolic distribution.
    """
    p, a, b, loc, scale = _validate_params(p, a, b, loc, scale)
    d = mp.sqrt((a + b)*(a - b))
    m = b / d * mp.besselk(p + 1, d) / mp.besselk(p, d)
    return loc + scale*m


@mp.extradps(5)
def var(p, a, b, loc=0, scale=1):
    """
    Variance of the generalized hyperbolic distribution.
    """
    p, a, b, loc, scale = _validate_params(p, a, b, loc, scale)
    t = (a + b)*(a - b)
    d = mp.sqrt(t)
    k0 = mp.besselk(p, d)
    k1 = mp.besselk(p + 1, d)
    k2 = mp.besselk(p + 2, d)
    return scale**2*(k1/k0/d + (b**2/t)*(k2/k0 - (k1/k0)**2))
