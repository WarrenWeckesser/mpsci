"""
Hyperbolic secant distribution
------------------------------

The wikipedia article

    https://en.wikipedia.org/wiki/Hyperbolic_secant_distribution

provides information about the distribution.  The scale parameter
used here matches that of SciPy's `scipy.special.hypsecant`; it is
2/pi times the scale parameter used in the wikipedia article.
"""

from mpmath import mp
from ._common import (_validate_loc_scale, _validate_p, _validate_x_bounds,
                      Initial)
from .. import stats


__all__ = ['pdf', 'logpdf', 'cdf', 'sf', 'invcdf', 'invsf',
           'support', 'mean', 'var', 'entropy', 'nll', 'mle']


def _validate_args(x, loc, scale):
    x = mp.mpf(x)
    loc, scale = _validate_loc_scale(loc, scale)
    return x, loc, scale


def pdf(x, loc=0, scale=1):
    """
    Probability density function for the hyperbolic secant distribution.
    """
    with mp.extradps(5):
        x, loc, scale = _validate_args(x, loc, scale)
        z = (x - loc)/scale
        return mp.sech(z)/mp.pi/scale


def logpdf(x, loc=0, scale=1):
    """
    Logarithm of the PDF of the hyperbolic secant distribution.
    """
    # This could be formulated many ways, but would any of them have
    # numerical advantages?
    return mp.log(pdf(x, loc, scale))


def cdf(x, loc=0, scale=1):
    """
    Cumulative distribution function for the hyperbolic secant distribution.
    """
    with mp.extradps(5):
        x, loc, scale = _validate_args(x, loc, scale)
        z = (x - loc)/scale
        return (2/mp.pi)*mp.atan(mp.exp(z))


def sf(x, loc=0, scale=1):
    """
    Survival function for the hyperbolic secant distribution.
    """
    with mp.extradps(5):
        x, loc, scale = _validate_args(x, loc, scale)
        z = (x - loc)/scale
        return (2/mp.pi)*mp.atan(mp.exp(-z))


def invcdf(p, loc=0, scale=1):
    """
    Inverse of the CDF for the hyperbolic secant distribution.

    This function is also known as the quantile function or the percent
    point function.
    """
    with mp.extradps(5):
        loc, scale = _validate_loc_scale(loc, scale)
        p = _validate_p(p)
        return loc + scale*mp.log(mp.tan(mp.pi*p/2))


def invsf(p, loc=0, scale=1):
    """
    Inverse of the survival function of the hyperblic secant distribution.
    """
    with mp.extradps(5):
        loc, scale = _validate_loc_scale(loc, scale)
        p = _validate_p(p)
        return loc - scale*mp.log(mp.tan(mp.pi*p/2))


def support(loc=0, scale=1):
    """
    Support of the hyperbolic secant distribution.
    """
    with mp.extradps(5):
        loc, scale = _validate_loc_scale(loc, scale)
        return (mp.ninf, mp.inf)


def mean(loc=0, scale=1):
    """
    Mean of the hyperbolic secant distribution.
    """
    with mp.extradps(5):
        loc, scale = _validate_loc_scale(loc, scale)
        return loc


def var(loc=0, scale=1):
    """
    Variance of the hyperbolic secant distribution.
    """
    with mp.extradps(5):
        loc, scale = _validate_loc_scale(loc, scale)
        return (mp.pi/2*scale)**2


def entropy(loc=0, scale=1):
    """
    Differential entropy of the hyperbolic secant distribution.
    """
    with mp.extradps(5):
        loc, scale = _validate_loc_scale(loc, scale)
        return mp.log(2*mp.pi) + mp.log(scale)


def nll(x, loc, scale):
    """
    Negative log-likelihood of the hyperbolic secant distribution.

    `x` must be a sequence of nonnegative numbers.
    """
    with mp.extradps(5):
        loc, scale = _validate_loc_scale(loc, scale)
        x = _validate_x_bounds(x)
        return -mp.fsum([logpdf(t, loc, scale) for t in x])


def mle(x, *, loc=None, scale=None):
    """
    Maximum likelihood estimate for the hyperbolic secant distribution.

    `x` must be a sequence of numbers.

    Provide a numerical argument to `loc` or `scale` to fix that
    parameter.  Provide an instance of `mpsci.distributions.Initial`
    to override the default initial value for the parmeter to be used
    in the numerical root-finding.
    """
    with mp.extradps(10):
        x = _validate_x_bounds(x)
        n = len(x)

        if ((loc is None or isinstance(loc, Initial)) and
                (scale is None or isinstance(scale, Initial))):
            # Fit both parameters.
            loc0 = stats.mean(x) if loc is None else mp.mpf(loc.initial)
            scale0 = stats.std(x) if scale is None else mp.mpf(scale.initial)

            def mle_eqns(loc, scale):
                eq1 = mp.fsum(mp.tanh((t - loc)/scale) for t in x)
                s2 = mp.fsum((t - loc)*mp.tanh((t - loc)/scale)/scale
                             for t in x)
                eq2 = s2 - n
                return eq1, eq2

            loc_hat, scale_hat = mp.findroot(mle_eqns, [loc0, scale0])
            return loc_hat, scale_hat

        if loc is None or isinstance(loc, Initial):
            # Fit loc only; scale is fixed.
            loc0 = stats.mean(x) if loc is None else mp.mpf(loc.initial)
            loc0, scale = _validate_loc_scale(loc0, scale)

            def mle_loc_eqn(loc):
                return mp.fsum(mp.tanh((t - loc)/scale) for t in x)

            loc_hat = mp.findroot(mle_loc_eqn, loc0)
            return loc_hat, scale

        if scale is None or isinstance(scale, Initial):
            # Fit scale; loc is fixed.
            loc = mp.mpf(loc)
            if scale is None:
                scale0 = stats.std([t - loc for t in x])
            else:
                scale0 = mp.mpf(scale.initial)
            loc, scale0 = _validate_loc_scale(loc, scale0)

            def mle_scale_eqn(scale):
                s = mp.fsum((t - loc)*mp.tanh((t - loc)/scale)/scale
                            for t in x)
                return s - n

            scale_hat = mp.findroot(mle_scale_eqn, scale0)
            return loc, scale_hat

        # Both parameters fixed, nothing to do.
        loc, scale = _validate_loc_scale(loc, scale)
        return loc, scale
