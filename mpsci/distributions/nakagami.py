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

import mpmath
from mpsci import stats


__all__ = ['pdf', 'logpdf', 'cdf', 'sf',
           'mean', 'var', 'nll', 'mle']


def _validate_params(nu, loc, scale):
    if nu <= 0:
        raise ValueError('nu must be positive')
    if scale <= 0:
        raise ValueError('sigma must be positive')


def pdf(x, nu, loc=0, scale=1):
    """
    Probability density function for the Nakagami distribution.
    """
    _validate_params(nu, loc, scale)
    with mpmath.extradps(5):
        if x <= loc:
            return mpmath.mp.zero
        x = mpmath.mpf(x)
        nu = mpmath.mpf(nu)
        loc = mpmath.mpf(loc)
        scale = mpmath.mpf(scale)
        z = (x - loc)/scale
        return (2*nu**nu * z**(2*nu - 1) * mpmath.exp(-nu*z**2)
                / mpmath.gamma(nu) / scale)


def logpdf(x, nu, loc=0, scale=1):
    """
    Natural logarithm of the PDF of the Nakagami distribution.
    """
    _validate_params(nu, loc, scale)
    if x <= 0:
        return -mpmath.inf
    with mpmath.extradps(5):
        if x <= loc:
            return mpmath.mp.zero
        x = mpmath.mpf(x)
        nu = mpmath.mpf(nu)
        loc = mpmath.mpf(loc)
        scale = mpmath.mpf(scale)
        z = (x - loc)/scale
        return (mpmath.log(2) + nu*mpmath.log(nu) - mpmath.loggamma(nu)
                + (2*nu-1)*mpmath.log(z) - nu*z**2 - mpmath.log(scale))


def cdf(x, nu, loc=0, scale=1):
    """
    Cumulative distribution function for the Nakagami distribution.
    """
    _validate_params(nu, loc, scale)
    if x <= 0:
        return mpmath.mp.zero

    with mpmath.extradps(5):
        if x <= loc:
            return mpmath.mp.zero
        x = mpmath.mpf(x)
        nu = mpmath.mpf(nu)
        loc = mpmath.mpf(loc)
        scale = mpmath.mpf(scale)
        z = (x - loc)/scale
        return mpmath.gammainc(nu, 0, nu*z**2, regularized=True)


def sf(x, nu, loc=0, scale=1):
    """
    Survival function for the Nakagami distribution.
    """
    _validate_params(nu, loc, scale)
    if x <= 0:
        return mpmath.mp.one

    with mpmath.extradps(5):
        if x <= loc:
            return mpmath.mp.zero
        x = mpmath.mpf(x)
        nu = mpmath.mpf(nu)
        loc = mpmath.mpf(loc)
        scale = mpmath.mpf(scale)
        z = (x - loc)/scale
        return mpmath.gammainc(nu, nu*z**2, mpmath.inf, regularized=True)


def mean(nu, loc=0, scale=1):
    """
    Mean of the Nakagami distribution.
    """
    _validate_params(nu, loc, scale)
    with mpmath.extradps(5):
        nu = mpmath.mpf(nu)
        loc = mpmath.mpf(loc)
        scale = mpmath.mpf(scale)
        gratio = mpmath.gammaprod([nu + 0.5], [nu])
        mean0 = gratio / mpmath.sqrt(nu)
        return loc + scale*mean0


def var(nu, loc=0, scale=1):
    """
    Variance of the Nakagami distribution.
    """
    _validate_params(nu, loc, scale)
    with mpmath.extradps(5):
        nu = mpmath.mpf(nu)
        scale = mpmath.mpf(scale)
        gratio = mpmath.gammaprod([nu + 0.5], [nu])
        var0 = 1 - gratio**2/nu
        return scale**2 * var0


def _validate_x(x, loc=0):
    if any(t <= loc for t in x):
        raise ValueError(f'All values in x must be greater than loc ({loc}).')


def nll(x, nu, loc, scale):
    """
    Negative log-likelihood function for the Nakagami distribution.
    """
    _validate_params(nu, loc, scale)
    _validate_x(x, loc=loc)
    n = len(x)
    with mpmath.extradps(5):
        nu = mpmath.mpf(nu)
        loc = mpmath.mpf(loc)
        scale = mpmath.mpf(scale)
        z = [(t - loc)/scale for t in x]
        logsum = mpmath.fsum([mpmath.log(t) for t in z])
        sqsum = mpmath.fsum([t**2 for t in z])
        ll = n*(mpmath.log(2) + nu*mpmath.log(nu) - mpmath.loggamma(nu)
                - mpmath.log(scale)) + (2*nu - 1)*logsum - nu*sqsum
        return -ll


def nll_grad(x, nu, loc, scale):
    """
    Gradient of the negative log-likelihood for the Nakagami distribution.
    """
    _validate_params(nu, loc, scale)
    _validate_x(x, loc=loc)
    n = len(x)
    with mpmath.extradps(5):
        nu = mpmath.mpf(nu)
        loc = mpmath.mpf(loc)
        scale = mpmath.mpf(scale)
        xloc = [(t - loc) for t in x]
        dldnu = (n*(1 + mpmath.log(nu) - mpmath.digamma(nu))
                 + 2*mpmath.fsum([mpmath.log(t/scale) for t in xloc])
                 - mpmath.fsum([t**2 for t in xloc])/scale**2)
        dldloc = -(2*nu - 1)*mpmath.fsum([1/t for t in xloc]) + 2*nu*mpmath.fsum(xloc)/scale**2
        dldscale = (2*nu/scale)*(-n + mpmath.fsum([t**2 for t in xloc])/scale**2)
        return dldnu, dldloc, dldscale


def _mle_nu_func(nu, scale, R):
    # This function is used in mle() to solve log(nu) - digamma(nu) = R.
    nu = mpmath.mpf(nu)
    return mpmath.log(nu) - mpmath.digamma(nu) - R


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


def mle(x, nu=None, loc=None, scale=None):
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
        _validate_x(x, loc)
        with mpmath.extradps(5):
            loc0 = mpmath.mpf(loc)
            x = [t - loc0 for t in x]
    else:
        raise ValueError('Fitting `loc` is not implemented yet. '
                         '`loc` must be given.')

    # If here, loc is fixed, and we've handled that by shifting x.
    # Either nu or scale (or both) are not fixed.

    if scale is None:
        scale0 = mpmath.sqrt(mpmath.fsum([t**2 for t in x])/len(x))
    else:
        scale0 = mpmath.mpf(scale)

    if nu is not None:
        nu0 = mpmath.mpf(nu)
    else:
        if scale is None:
            R = -stats.mean([2*mpmath.log(t/scale0) for t in x])
        else:
            R = (stats.mean([(t/scale0)**2 for t in x]) -
                 stats.mean([2*mpmath.log(t/scale0) for t in x]) - 1)
        nu0 = _estimate_nu(R)
        nu0 = mpmath.findroot(lambda nu: _mle_nu_func(nu, scale0, R), nu0)

    return nu0, loc0, scale0
