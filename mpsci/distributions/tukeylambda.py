"""
Tukey's Lambda distribution
---------------------------

See https://en.wikipedia.org/wiki/Tukey_lambda_distribution

This implementation uses the same parameter names as
``scipy.stats.tukeylambda``.

"""

from mpmath import mp, findroot
from ._common import _validate_loc_scale, _validate_p


__all__ = ['pdf', 'logpdf', 'cdf', 'sf', 'invcdf', 'invsf',
           'support', 'mean', 'var']


def _validate_params(lam, loc, scale):
    lam = mp.mpf(lam)
    loc, scale = _validate_loc_scale(loc, scale)
    return lam, loc, scale


@mp.extradps(5)
def pdf(x, lam, loc=0, scale=1):
    """
    Probability density function for Tukey's lambda distribution.
    """
    lam, loc, scale = _validate_params(lam, loc, scale)
    z = (x - loc)/scale
    if lam > 0:
        lamrecip = 1/lam
        if z < -lamrecip or z > lamrecip:
            return mp.zero
    c = cdf(z, lam)
    p = 1/(mp.power(c, lam - 1) + mp.exp((lam - 1)*mp.log1p(-c)))
    return p/scale


@mp.extradps(5)
def logpdf(x, lam, loc=0, scale=1):
    """
    Natural logarithm of the PDF for Tukey's lambda distribution.

    This implementation is simply log(pdf(x, lam, loc, scale)).
    """
    lam, loc, scale = _validate_params(lam, loc, scale)
    # TODO: Use a formulation that provides better precision.
    return mp.log(pdf(x, lam, loc, scale))


def _cdf_solver_bracket(z, lam):
    """
    Get a bracket for p for solving invcdf(p, lam) = z.
    """
    lamrecip = 1/lam
    if lam < 0:
        if z < 0:
            if z < mp.power(2.0, -lam)/lam:
                pmax = mp.power(lam*z, lamrecip)
            else:
                pmax = 0.5
            if z < -mp.power(2.0, -lam - 1):
                pmin = 0.0
            else:
                pmin = mp.one/2 + mp.power(2, lam)*z
        else:
            # z > 0
            if z > -mp.power(2.0, -lam)/lam:
                pmin = -mp.expm1(mp.log(-lam*z)/lam)
            else:
                pmin = 0.5
            if z > mp.power(2.0, -lam - 1):
                pmax = 1.0
            else:
                pmax = mp.one/2 + mp.power(2, lam)*z
    else:
        pmin = 0.0
        pmax = 1.0
    return (pmin, pmax)


@mp.extradps(5)
def cdf(x, lam, loc=0, scale=1, solver='ridder'):
    """
    Cumulative distribution function for Tukey's lambda distribution.

    This function uses `mpmath.findroot` to compute the survival
    function.

    `solver` is passed to `mpmath.findroot`.  It can be one of
    ['bisect', 'anderson', 'ridder'].
    """
    lam, loc, scale = _validate_params(lam, loc, scale)

    z = (x - loc)/scale
    if lam > 0:
        lamrecip = 1/lam
        if z <= -lamrecip:
            return mp.zero
        if z >= lamrecip:
            return mp.one

    if lam == 0:
        return mp.one/(mp.exp(-z) + 1)

    # Get starting bracket for bisection.
    pmin, pmax = _cdf_solver_bracket(z, lam)

    p = findroot(lambda t: invcdf(t, lam) - z, [pmin, pmax],
                 solver=solver)
    return p


@mp.extradps(5)
def sf(x, lam, loc=0, scale=1, solver='ridder'):
    """
    Survival function for Tukey's lambda distribution.

    This function uses `mpmath.findroot` to compute the survival
    function.

    `solver` is passed to `mpmath.findroot`.  It can be one of
    ['bisect', 'anderson', 'ridder'].
    """
    lam, loc, scale = _validate_params(lam, loc, scale)

    z = (x - loc)/scale
    if lam > 0:
        lamrecip = 1/lam
        if z <= -lamrecip:
            return mp.one
        if z >= lamrecip:
            return mp.zero

    if lam == 0:
        ez = mp.exp(-z)
        return ez/(ez + 1)

    # Get starting bracket for bisection.
    pmin, pmax = _cdf_solver_bracket(-z, lam)

    p = findroot(lambda t: invsf(t, lam) - z, [pmin, pmax],
                 solver=solver)
    return p


@mp.extradps(5)
def invcdf(p, lam, loc=0, scale=1):
    """
    Quantile function for Tukey's lambda distribution.
    """
    p = _validate_p(p)
    lam, loc, scale = _validate_params(lam, loc, scale)

    if lam < 0:
        if p == 0:
            return mp.ninf
        if p == 1:
            return mp.inf

    if lam == 0.0:
        z = mp.log(p) - mp.log1p(-p)
    else:
        z = (p**lam - mp.exp(lam*mp.log1p(-p)))/lam
    return loc + scale*z


@mp.extradps(5)
def invsf(p, lam, loc=0, scale=1):
    """
    Inverse survival function for Tukey's lambda distribution.
    """
    p = _validate_p(p)
    lam, loc, scale = _validate_params(lam, loc, scale)

    if lam < 0:
        if p == 0:
            return mp.inf
        if p == 1:
            return mp.ninf

    if lam == 0.0:
        z = mp.log1p(-p) - mp.log(p)
    else:
        z = (mp.exp(lam*mp.log1p(-p)) - p**lam)/lam
    return loc + scale*z


@mp.extradps(5)
def support(lam, loc=0, scale=1):
    """
    Support interval for Tukey's lambda distribution.
    """
    lam, loc, scale = _validate_params(lam, loc, scale)
    if lam <= 0:
        return (mp.ninf, mp.inf)
    else:
        return (loc - scale/lam, loc + scale/lam)


@mp.extradps(5)
def mean(lam, loc=0, scale=1):
    """
    Mean of Tukey's lambda distribution.

    The mean is `loc` if `lam` > -1.  Otherwise the mean is not defined,
    so `nan` is returned.
    """
    lam, loc, scale = _validate_params(lam, loc, scale)
    if lam > -1:
        return loc
    return mp.nan


@mp.extradps(5)
def var(lam, loc=0, scale=1):
    """
    Variance of Tukey's lambda distribution.

    `nan` is returned if `lam` <= -1/2.
    """
    lam, loc, scale = _validate_params(lam, loc, scale)
    if lam <= -0.5:
        return mp.nan
    if lam == 0:
        return (scale*mp.pi)**2/3
    g1 = mp.gamma(lam + 1)
    g2 = mp.gamma(2*lam + 2)
    v0 = (2/lam**2)*(1/(1 + 2*lam) - g1**2/g2)
    return scale**2 * v0
