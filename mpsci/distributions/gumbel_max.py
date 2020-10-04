"""
Gumbel probability distribution (for maxima)
--------------------------------------------

This is the same distribution as `scipy.stats.gumbel_r`.
"""

import mpmath
from .. import stats


__all__ = ['pdf', 'logpdf', 'cdf', 'invcdf', 'sf', 'invsf', 'mean', 'var',
           'nll', 'mle']


def pdf(x, loc, scale):
    """
    Probability density function for the Gumbel distribution (for maxima).
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
        return -(z + mpmath.exp(-z)) - mpmath.log(scale)


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
        return mpmath.exp(-mpmath.exp(-z))


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
        z = -mpmath.log(-mpmath.log(p))
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
        return -mpmath.expm1(-mpmath.exp(-z))


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
        z = -mpmath.log(-mpmath.log1p(-p))
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
        return loc + mpmath.euler*scale


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
        t2 = mpmath.fsum(z)
        t3 = mpmath.fsum([mpmath.exp(-zi) for zi in z])
        return t1 + t2 + t3


def _mle_scale_func(scale, x, xbar):
    emx = [mpmath.exp(-xi/scale) for xi in x]
    s1 = mpmath.fsum([xi * emxi for xi, emxi in zip(x, emx)])
    s2 = mpmath.fsum(emx)
    return s2*(xbar - scale) - s1


def _mle_scale_with_fixed_loc(scale, x, loc):
    z = [(xi - loc) / scale for xi in x]
    ez = [mpmath.expm1(-zi)*zi for zi in z]
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

    >>> import mpmath
    >>> mpmath.mp.dps = 20
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
            xbar = stats.mean(x)
            s = stats.std(x)  # Initial guess for the scale.
            scale = mpmath.findroot(lambda t: _mle_scale_func(t, x, xbar),
                                    s)
        else:
            scale = mpmath.mpf(scale)
        if loc is None:
            ex = [mpmath.exp(-xi / scale) for xi in x]
            loc = -scale * mpmath.log(stats.mean(ex))
        else:
            loc = mpmath.mpf(loc)

        return loc, scale
