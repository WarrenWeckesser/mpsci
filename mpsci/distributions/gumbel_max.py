"""
Gumbel probability distribution (for maxima)
--------------------------------------------

This is the same distribution as:

* `scipy.stats.gumbel_r`;
* NumPy's `numpy.random.Generator.gumbel`;
* the Gumbel distribution discussed in the wikipedia article
  "Gumbel distribtion" (https://en.wikipedia.org/wiki/Gumbel_distribution);
* the Type I extreme value distribution used in the text "An Introduction
  to Statistical Modeling of Extreme Values" by Stuart Coles (Springer, 2001);
* the Gumbel distribution given in the text "Modelling Extremal Events" by
  Embrechts, Klüppelberg and Mikosch (Springer, 1997);
* the Gumbel distribution in the text "Statistical Distribution" (fourth ed.)
  by Forbes, Evans, Hastings and Peacock (Wiley, 2011).

"""

from mpmath import mp
from .. import stats
from mpsci.stats import mean as _mean


__all__ = ['pdf', 'logpdf', 'cdf', 'invcdf', 'sf', 'invsf', 'mean', 'var',
           'nll', 'mle', 'mom']


def pdf(x, loc, scale):
    """
    Probability density function for the Gumbel distribution (for maxima).
    """
    if scale <= 0:
        raise ValueError('scale must be positive.')

    with mp.extradps(5):
        x = mp.mpf(x)
        loc = mp.mpf(loc)
        scale = mp.mpf(scale)
        return mp.exp(logpdf(x, loc, scale))


def logpdf(x, loc, scale):
    """
    Log of the PDF of the Gumbel distribution.
    """
    if scale <= 0:
        raise ValueError('scale must be positive.')

    with mp.extradps(5):
        x = mp.mpf(x)
        loc = mp.mpf(loc)
        scale = mp.mpf(scale)
        z = (x - loc) / scale
        return -(z + mp.exp(-z)) - mp.log(scale)


def cdf(x, loc, scale):
    """
    Cumulative distribution function for the Gumbel distribution.
    """
    if scale <= 0:
        raise ValueError('scale must be positive.')

    with mp.extradps(5):
        x = mp.mpf(x)
        loc = mp.mpf(loc)
        scale = mp.mpf(scale)
        z = (x - loc) / scale
        return mp.exp(-mp.exp(-z))


def invcdf(p, loc, scale):
    """
    Inverse of the CDF for the Gumbel distribution.
    """
    if scale <= 0:
        raise ValueError('scale must be positive.')

    with mp.extradps(5):
        p = mp.mpf(p)
        loc = mp.mpf(loc)
        scale = mp.mpf(scale)
        z = -mp.log(-mp.log(p))
        x = scale*z + loc
        return x


def sf(x, loc, scale):
    """
    Survival function for the Gumbel distribution.
    """
    if scale <= 0:
        raise ValueError('scale must be positive.')

    with mp.extradps(5):
        x = mp.mpf(x)
        loc = mp.mpf(loc)
        scale = mp.mpf(scale)
        z = (x - loc) / scale
        return -mp.expm1(-mp.exp(-z))


def invsf(p, loc, scale):
    """
    Inverse of the survival function for the Gumbel distribution.
    """
    if scale <= 0:
        raise ValueError('scale must be positive.')

    with mp.extradps(5):
        p = mp.mpf(p)
        loc = mp.mpf(loc)
        scale = mp.mpf(scale)
        z = -mp.log(-mp.log1p(-p))
        x = scale*z + loc
        return x


def mean(loc, scale):
    """
    Mean of the Gumbel distribution.
    """
    if scale <= 0:
        raise ValueError('scale must be positive.')

    with mp.extradps(5):
        loc = mp.mpf(loc)
        scale = mp.mpf(scale)
        return loc + mp.euler*scale


def var(loc, scale):
    """
    Variance of the Gumbel distribution.
    """
    if scale <= 0:
        raise ValueError('scale must be positive.')

    with mp.extradps(5):
        loc = mp.mpf(loc)
        scale = mp.mpf(scale)
        return mp.pi**2/6 * scale**2


def nll(x, loc, scale):
    """
    Negative log-likelihood function for the Gumbel distribution.
    """
    if scale <= 0:
        raise ValueError('scale must be positive.')

    with mp.extradps(5):
        loc = mp.mpf(loc)
        scale = mp.mpf(scale)
        n = len(x)
        z = [(mp.mpf(xi) - loc)/scale for xi in x]
        t1 = n*mp.log(scale)
        t2 = mp.fsum(z)
        t3 = mp.fsum([mp.exp(-zi) for zi in z])
        return t1 + t2 + t3


def _mle_scale_func(scale, x, xbar):
    emx = [mp.exp(-xi/scale) for xi in x]
    s1 = mp.fsum([xi * emxi for xi, emxi in zip(x, emx)])
    s2 = mp.fsum(emx)
    return s2*(xbar - scale) - s1


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
    sign2 = mp.sign(_mle_scale_func(s2, x, xbar))
    while True:
        s1 = 0.9*s1
        sign1 = mp.sign(_mle_scale_func(s1, x, xbar))
        if (sign1 * sign2) <= 0:
            break
        s2 = 1.1*s2
        sign2 = mp.sign(_mle_scale_func(s2, x, xbar))
        if (sign1 * sign2) <= 0:
            break

    # Did we stumble across the root while looking for an interval
    # with a sign change?  Not likely, but check anyway...
    if sign1 == 0:
        return s1
    if sign2 == 0:
        return s2

    root = mp.findroot(lambda t: _mle_scale_func(t, x, xbar),
                       [s1, s2], solver='anderson')

    return root


def _mle_scale_with_fixed_loc(scale, x, loc):
    z = [(xi - loc) / scale for xi in x]
    ez = [mp.expm1(-zi)*zi for zi in z]
    return stats.mean(ez) + 1


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

    >>> from mpmath import mp
    >>> mp.dps = 20
    >>> from mpsci.distributions import gumbel_max

    The data to be fit:

    >>> x = [6.86, 14.8 , 15.65,  8.72,  8.11,  8.15, 13.01, 13.36]

    Unconstrained MLE:

    >>> gumbel_max.mle(x)
    (mpf('9.4879877926148360358863'), mpf('2.727868138859403832702'))

    If we know the scale is 2, we can add the argument `scale=2`:

    >>> gumbel_max.mle(x, scale=2)
    (mpf('9.1305625326153555632872'), mpf('2.0'))
    """
    with mp.extradps(5):
        x = [mp.mpf(xi) for xi in x]

        if scale is None and loc is not None:
            # Estimate scale with fixed loc.
            loc = mp.mpf(loc)
            # Initial guess for findroot.
            s0 = stats.std([xi - loc for xi in x])
            scale = mp.findroot(
                lambda t: _mle_scale_with_fixed_loc(t, x, loc), s0
            )
            return loc, scale

        if scale is None:
            scale = _solve_mle_scale(x)
        else:
            scale = mp.mpf(scale)

        if loc is None:
            ex = [mp.exp(-xi / scale) for xi in x]
            loc = -scale * mp.log(stats.mean(ex))
        else:
            loc = mp.mpf(loc)

        return loc, scale


def mom(x):
    """
    Method of moments parameter estimation for the Gumbel-max distribution.

    x must be a sequence of real numbers.

    Returns (loc, scale).
    """
    with mp.extradps(5):
        M1 = _mean(x)
        M2 = _mean([mp.mpf(t)**2 for t in x])
        scale = mp.sqrt(6*(M2 - M1**2))/mp.pi
        loc = M1 - scale*mp.euler
        return loc, scale
