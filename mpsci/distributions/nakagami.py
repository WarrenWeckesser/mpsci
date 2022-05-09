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


def _mle_nu_func(nu, scale, R):
    # This function is used in mle() to solve log(nu) - digamma(nu) = R.
    nu = mpmath.mpf(nu)
    return mpmath.log(nu) - mpmath.digamma(nu) - R


def _mle_scale_func(scale, x):
    # This function is used in mle() to find the MLE for the scale.
    # The equation to be solved is
    #     sum((x_i/scale)**2) = N
    # where N is len(x).
    scale = mpmath.mpf(scale)
    N = len(x)
    S2 = sum((t/scale)**2 for t in x)
    return S2 - N


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
    Nakagami distribution maximum likelihood parameter estimation.

    x must be a sequence of numbers.

    Returns (nu, loc, scale).

    Currently a fixed loc *must* be given.
    """
    if loc is not None:
        _validate_x(x, loc)
        x = [t - loc for t in x]
    else:
        raise ValueError('Fitting `loc` is not implemented yet. '
                         '`loc` must be given.')

    if scale is None:
        scale = mpmath.findroot(lambda scale: _mle_scale_func(scale, x),
                                stats.std(x))

    if nu is None:
        R = (stats.mean([(t/scale)**2 for t in x]) -
             stats.mean([2*mpmath.log(t/scale) for t in x]) - 1)
        nu0 = _estimate_nu(R)
        nu = mpmath.findroot(lambda nu: _mle_nu_func(nu, scale, R), nu0)

    return nu, loc, scale
