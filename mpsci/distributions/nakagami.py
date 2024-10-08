"""
Nakagami distribution
---------------------

See https://en.wikipedia.org/wiki/Nakagami_distribution.

The parametrization used here matches that of SciPy:

* ν > 0 is the shape parameter (same as m in the wikipedia article).
* ``loc`` is the location parameter.  (The wikipedia article does not
  include a location parameter for the distribution.)
* ``scale`` is the scale parameter. (``scale`` = sqrt(Ω), where Ω
  is the "spread" parameter used in the wikipedia article.

When ν = 1/2, the distribution is a half-normal distribution.  For
ν < 1/2, the PDF is 0 at x = 0; for ν > 1/2, the PDF approaches
infinity as x approaches 0.

"""

from mpmath import mp
from mpsci import stats
from ._common import _validate_x_bounds, _validate_p,  _find_bracket


__all__ = ['pdf', 'logpdf', 'cdf', 'invcdf', 'sf', 'invsf',
           'support', 'mean', 'var', 'entropy',
           'nll', 'mle']


def _validate_params(nu, loc=0, scale=1):
    if nu <= 0:
        raise ValueError('nu must be positive')
    if scale <= 0:
        raise ValueError('scale must be positive')
    return mp.mpf(nu), mp.mpf(loc), mp.mpf(scale)


@mp.extradps(5)
def pdf(x, nu, loc=0, scale=1):
    """
    Probability density function for the Nakagami distribution.
    """
    nu, loc, scale = _validate_params(nu, loc, scale)
    if x <= loc:
        return mp.zero
    x = mp.mpf(x)
    z = (x - loc)/scale
    return (2*nu**nu * z**(2*nu - 1) * mp.exp(-nu*z**2)
            / mp.gamma(nu) / scale)


@mp.extradps(5)
def logpdf(x, nu, loc=0, scale=1):
    """
    Natural logarithm of the PDF of the Nakagami distribution.
    """
    nu, loc, scale = _validate_params(nu, loc, scale)
    if x <= loc:
        return mp.ninf
    x = mp.mpf(x)
    z = (x - loc)/scale
    return (mp.log(2) + nu*mp.log(nu) - mp.loggamma(nu)
            + (2*nu-1)*mp.log(z) - nu*z**2 - mp.log(scale))


@mp.extradps(5)
def cdf(x, nu, loc=0, scale=1):
    """
    Cumulative distribution function for the Nakagami distribution.
    """
    nu, loc, scale = _validate_params(nu, loc, scale)
    if x <= loc:
        return mp.zero
    x = mp.mpf(x)
    z = (x - loc)/scale
    return mp.gammainc(nu, 0, nu*z**2, regularized=True)


@mp.extradps(5)
def invcdf(p, nu, loc=0, scale=1):
    """
    Inverse of the CDF of the Nakagmi distribution.

    The implementation uses a numerical root finder, so it may be slow, and
    it may fail to converge for some inputs.
    """
    p = _validate_p(p)
    nu, loc, scale = _validate_params(nu, loc, scale)
    with mp.extradps(mp.dps):
        x0, x1 = _find_bracket(lambda x: cdf(x, nu, loc, scale), p, 0, mp.inf)
        root = mp.findroot(lambda t: cdf(t, nu, loc, scale) - p, x0=(x0, x1))
    return root


@mp.extradps(5)
def sf(x, nu, loc=0, scale=1):
    """
    Survival function for the Nakagami distribution.
    """
    nu, loc, scale = _validate_params(nu, loc, scale)
    if x <= loc:
        return mp.one
    x = mp.mpf(x)
    z = (x - loc)/scale
    return mp.gammainc(nu, nu*z**2, mp.inf, regularized=True)


@mp.extradps(5)
def invsf(p, nu, loc=0, scale=1):
    """
    Inverse of the survival function of the Nakagmi distribution.

    The implementation uses a numerical root finder, so it may be slow, and
    it may fail to converge for some inputs.
    """
    p = _validate_p(p)
    nu, loc, scale = _validate_params(nu, loc, scale)
    with mp.extradps(mp.dps):
        x0, x1 = _find_bracket(lambda x: sf(x, nu, loc, scale), p, 0, mp.inf)
        root = mp.findroot(lambda t: sf(t, nu, loc, scale) - p, x0=(x0, x1))
    return root


@mp.extradps(5)
def support(nu, loc=0, scale=1):
    """
    Support of the Nakagami distribution.
    """
    nu, loc, scale = _validate_params(nu, loc, scale)
    return (loc, mp.inf)


@mp.extradps(5)
def mean(nu, loc=0, scale=1):
    """
    Mean of the Nakagami distribution.
    """
    nu, loc, scale = _validate_params(nu, loc, scale)
    gratio = mp.gammaprod([nu + 0.5], [nu])
    mean0 = gratio / mp.sqrt(nu)
    return loc + scale*mean0


@mp.extradps(5)
def var(nu, loc=0, scale=1):
    """
    Variance of the Nakagami distribution.
    """
    nu, loc, scale = _validate_params(nu, loc, scale)
    gratio = mp.gammaprod([nu + 0.5], [nu])
    var0 = 1 - gratio**2/nu
    return scale**2 * var0


@mp.extradps(5)
def entropy(nu, loc=0, scale=1):
    """
    Differential entropy of the Nakagami distribution.
    """
    nu, loc, scale = _validate_params(nu, loc, scale)
    return ((mp.one/2 - nu)*mp.digamma(nu) - mp.log(nu)/2
            + nu + mp.loggamma(nu) - mp.log(2) + mp.log(scale))


@mp.extradps(5)
def nll(x, nu, loc, scale):
    """
    Negative log-likelihood function for the Nakagami distribution.
    """
    n = len(x)
    nu, loc, scale = _validate_params(nu, loc, scale)
    x = _validate_x_bounds(x, low=loc, strict_low=True, lowname='loc')
    z = [(t - loc)/scale for t in x]
    logsum = mp.fsum([mp.log(t) for t in z])
    sqsum = mp.fsum([t**2 for t in z])
    ll = n*(mp.log(2) + nu*mp.log(nu) - mp.loggamma(nu)
            - mp.log(scale)) + (2*nu - 1)*logsum - nu*sqsum
    return -ll


@mp.extradps(5)
def nll_grad(x, nu, loc, scale):
    """
    Gradient of the negative log-likelihood for the Nakagami distribution.
    """
    n = len(x)
    nu, loc, scale = _validate_params(nu, loc, scale)
    x = _validate_x_bounds(x, low=loc, strict_low=True, lowname='loc')
    xloc = [(t - loc) for t in x]
    dldnu = (n*(1 + mp.log(nu) - mp.digamma(nu))
             + 2*mp.fsum([mp.log(t/scale) for t in xloc])
             - mp.fsum([t**2 for t in xloc])/scale**2)
    sum_inv = mp.fsum([1/t for t in xloc])
    dldloc = -(2*nu - 1)*sum_inv + 2*nu*mp.fsum(xloc)/scale**2
    sum_sq = mp.fsum([t**2 for t in xloc])
    dldscale = (2*nu/scale)*(-n + sum_sq/scale**2)
    return -dldnu, -dldloc, -dldscale


def _mle_nu_func(nu, R):
    # This function is used in mle() to solve log(nu) - digamma(nu) = R.
    nu = mp.mpf(nu)
    return mp.log(nu) - mp.digamma(nu) - R


def _estimate_nu(R):
    """
    Estimate the solution of log(nu) - psi(nu) = R.
    """
    if R >= 10:
        return 1/R
    elif R < 5e-1:
        return 1/(2*R)
    else:
        return 1/(1.5*R)


@mp.extradps(5)
def mle(x, *, nu=None, loc=None, scale=None):
    """
    Maximum likelihood parameter estimation for the Nakagami distribution.

    x must be a sequence of numbers.

    Returns (nu, loc, scale).

    Currently a fixed loc *must* be given.
    """
    if nu is not None and loc is not None and scale is not None:
        # Nothing to do.
        return nu, loc, scale

    if loc is not None:
        # loc is fixed; handle this by subtracting loc from x.
        with mp.extradps(5):
            x = _validate_x_bounds(x, low=loc, strict_low=True, lowname='loc')
            loc0 = mp.mpf(loc)
            x = [t - loc0 for t in x]
    else:
        raise ValueError('Fitting `loc` is not implemented yet. '
                         '`loc` must be given.  All values in `x` must'
                         'be strictly greater than `loc`.')

    # If here, loc is fixed, and we've handled that by shifting x.
    # Either nu or scale (or both) are not fixed.

    if scale is None:
        scale0 = mp.sqrt(mp.fsum([t**2 for t in x])/len(x))
    else:
        scale0 = mp.mpf(scale)

    if nu is not None:
        nu0 = mp.mpf(nu)
    else:
        if scale is None:
            R = -stats.mean([2*mp.log(t/scale0) for t in x])
        else:
            R = (stats.mean([(t/scale0)**2 for t in x]) -
                 stats.mean([2*mp.log(t/scale0) for t in x]) - 1)
        nu0 = _estimate_nu(R)
        nu0 = mp.findroot(lambda nu: _mle_nu_func(nu, R), nu0)

    return nu0, loc0, scale0
