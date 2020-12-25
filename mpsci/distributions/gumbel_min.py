"""
Gumbel probability distribution (for minima)
--------------------------------------------

This is the same distribution as:

* `scipy.stats.gumbel_l`;
* Wolfram Alpha's `GumbelDistribution`.

"""

import mpmath
from .. import stats
from mpsci.stats import mean as _mean


__all__ = ['pdf', 'logpdf', 'cdf', 'invcdf', 'sf', 'invsf', 'mean', 'var',
           'nll', 'mle', 'mom']


def pdf(x, loc, scale):
    """
    Probability density function for the Gumbel distribution (for minima).
    """
    if scale <= 0:
        raise ValueError('scale must be positive.')

    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        loc = mpmath.mpf(loc)
        scale = mpmath.mpf(scale)
        return mpmath.exp(logpdf(x, loc, scale))


def logpdf(x, loc, scale):
    """
    Log of the PDF of the Gumbel distribution.
    """
    if scale <= 0:
        raise ValueError('scale must be positive.')

    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        loc = mpmath.mpf(loc)
        scale = mpmath.mpf(scale)
        z = (x - loc) / scale
        return -(-z + mpmath.exp(z)) - mpmath.log(scale)


def cdf(x, loc, scale):
    """
    Cumulative distribution function for the Gumbel distribution.
    """
    if scale <= 0:
        raise ValueError('scale must be positive.')

    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        loc = mpmath.mpf(loc)
        scale = mpmath.mpf(scale)
        z = (x - loc) / scale
        return -mpmath.expm1(-mpmath.exp(z))


def invcdf(p, loc, scale):
    """
    Inverse of the CDF for the Gumbel distribution.
    """
    if scale <= 0:
        raise ValueError('scale must be positive.')

    with mpmath.extradps(5):
        p = mpmath.mpf(p)
        loc = mpmath.mpf(loc)
        scale = mpmath.mpf(scale)
        z = mpmath.log(-mpmath.log1p(-p))
        x = scale*z + loc
        return x


def sf(x, loc, scale):
    """
    Survival function for the Gumbel distribution.
    """
    if scale <= 0:
        raise ValueError('scale must be positive.')

    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        loc = mpmath.mpf(loc)
        scale = mpmath.mpf(scale)
        z = (x - loc) / scale
        return mpmath.exp(-mpmath.exp(z))


def invsf(p, loc, scale):
    """
    Inverse of the survival function for the Gumbel distribution.
    """
    if scale <= 0:
        raise ValueError('scale must be positive.')

    with mpmath.extradps(5):
        p = mpmath.mpf(p)
        loc = mpmath.mpf(loc)
        scale = mpmath.mpf(scale)
        z = mpmath.log(-mpmath.log(p))
        x = scale*z + loc
        return x


def mean(loc, scale):
    """
    Mean of the Gumbel distribution.
    """
    if scale <= 0:
        raise ValueError('scale must be positive.')

    with mpmath.extradps(5):
        loc = mpmath.mpf(loc)
        scale = mpmath.mpf(scale)
        return loc - mpmath.euler*scale


def var(loc, scale):
    """
    Variance of the Gumbel distribution.
    """
    if scale <= 0:
        raise ValueError('scale must be positive.')

    with mpmath.extradps(5):
        loc = mpmath.mpf(loc)
        scale = mpmath.mpf(scale)
        return mpmath.pi**2/6 * scale**2


def nll(x, loc, scale):
    """
    Negative log-likelihood function for the Gumbel distribution.
    """
    if scale <= 0:
        raise ValueError('scale must be positive.')

    with mpmath.extradps(5):
        loc = mpmath.mpf(loc)
        scale = mpmath.mpf(scale)
        n = len(x)
        z = [(mpmath.mpf(xi) - loc)/scale for xi in x]
        t1 = n*mpmath.log(scale)
        t2 = -mpmath.fsum(z)
        t3 = mpmath.fsum([mpmath.exp(zi) for zi in z])
        return t1 + t2 + t3


def _mle_scale_func(scale, x, xbar):
    emx = [mpmath.exp(xi/scale) for xi in x]
    s1 = mpmath.fsum([xi * emxi for xi, emxi in zip(x, emx)])
    s2 = mpmath.fsum(emx)
    return s2*(xbar + scale) - s1


def _solve_mle_scale(x):
    xbar = stats.mean(x)

    # Very rough guess of the scale parameter:
    s0 = stats.std(x)
    if s0 == 0:
        # The x values are all the same.
        return s0

    # Find an interval in which there is a sign change of
    # gumbel_min._mle_scale_func.
    s1 = s0
    s2 = s0
    sign2 = mpmath.sign(_mle_scale_func(s2, x, xbar))
    while True:
        s1 = 0.9*s1
        sign1 = mpmath.sign(_mle_scale_func(s1, x, xbar))
        if (sign1 * sign2) <= 0:
            break
        s2 = 1.1*s2
        sign2 = mpmath.sign(_mle_scale_func(s2, x, xbar))
        if (sign1 * sign2) <= 0:
            break

    # Did we stumble across the root while looking for an interval
    # with a sign change?  Not likely, but check anyway...
    if sign1 == 0:
        return s1
    if sign2 == 0:
        return s2

    root = mpmath.findroot(lambda t: _mle_scale_func(t, x, xbar),
                           [s1, s2], solver='anderson')

    return root


def _mle_scale_with_fixed_loc(scale, x, loc):
    z = [(xi - loc) / scale for xi in x]
    ez = [mpmath.expm1(zi)*zi for zi in z]
    return 1 - stats.mean(ez)


def mle(x, loc=None, scale=None):
    """
    Maximum likelihood estimates for the Gumbel distribution.

    `x` must be a sequence of numbers--it is the data to which the
    Gumbel distribution is to be fit.

    If either `loc` or `scale` is not None, the parameter is fixed
    at the given value, and only the other parameter will be fit.

    Returns maximum likelihood estimates of the `loc` and `scale`
    parameters.

    Examples
    --------
    Imports and mpmath configuration:

    >>> import mpmath
    >>> mpmath.mp.dps = 20
    >>> from mpsci.distributions import gumbel_min

    The data to be fit:

    >>> x = [6.86, 14.8 , 15.65,  8.72,  8.11,  8.15, 13.01, 13.36]

    Unconstrained MLE:

    >>> gumbel_min.mle(x)
    (mpf('12.708439639698245696235'), mpf('2.878444823276260896075'))

    If we know the scale is 2, we can add the argument `scale=2`:

    >>> gumbel_min.mle(x, scale=2)
    (mpf('13.18226169025112165358'), mpf('2.0'))
    """
    with mpmath.extradps(5):
        x = [mpmath.mpf(xi) for xi in x]

        if scale is None and loc is not None:
            # Estimate scale with fixed loc.
            loc = mpmath.mpf(loc)
            # Initial guess for findroot.
            s0 = stats.std([xi - loc for xi in x])
            scale = mpmath.findroot(
                lambda t: _mle_scale_with_fixed_loc(t, x, loc), s0
            )
            return loc, scale

        if scale is None:
            scale = _solve_mle_scale(x)
        else:
            scale = mpmath.mpf(scale)

        if loc is None:
            ex = [mpmath.exp(xi / scale) for xi in x]
            loc = scale * mpmath.log(stats.mean(ex))
        else:
            loc = mpmath.mpf(loc)

        return loc, scale


def mom(x):
    """
    Method of moments parameter estimation for the Gumbel-min distribution.

    x must be a sequence of real numbers.

    Returns (loc, scale).
    """
    with mpmath.extradps(5):
        M1 = _mean(x)
        M2 = _mean([mpmath.mpf(t)**2 for t in x])
        scale = mpmath.sqrt(6*(M2 - M1**2))/mpmath.pi
        loc = M1 + scale*mpmath.euler
        return loc, scale
