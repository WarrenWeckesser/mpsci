"""
Laplace distribution
--------------------


The parameters loc and scale correspond to the parameters mu and b of the
wikipedia article:

    https://en.wikipedia.org/wiki/Laplace_distribution

"""

from mpmath import mp
from mpsci.stats import mean as _mean
from ._common import _seq_to_mp, _validate_loc_scale


__all__ = ['pdf', 'logpdf', 'cdf', 'sf', 'invcdf', 'invsf', 'interval_prob',
           'support', 'mean', 'mode', 'var', 'mle', 'mom']


@mp.extradps(5)
def pdf(x, loc=0, scale=1):
    """
    Laplace distribution probability density function.
    """
    loc, scale = _validate_loc_scale(loc, scale)
    x = mp.mpf(x)
    return mp.exp(-abs(x - loc)/scale)/(2*scale)


@mp.extradps(5)
def logpdf(x, loc=0, scale=1):
    """
    Log of the PDF of the Laplace distribution.
    """
    loc, scale = _validate_loc_scale(loc, scale)
    x = mp.mpf(x)
    return -abs(x - loc)/scale - mp.log(2*scale)


@mp.extradps(5)
def cdf(x, loc=0, scale=1):
    """
    Laplace distribution cumulative distribution function.
    """
    loc, scale = _validate_loc_scale(loc, scale)
    x = mp.mpf(x)
    z = (x - loc) / scale
    if x <= loc:
        c = mp.exp(z)/2
    else:
        c = 1 - mp.exp(-z)/2
    return c


@mp.extradps(5)
def sf(x, loc=0, scale=1):
    """
    Laplace distribution survival function.
    """
    loc, scale = _validate_loc_scale(loc, scale)
    x = mp.mpf(x)
    z = (x - loc) / scale
    if x <= loc:
        c = 1 - mp.exp(z)/2
    else:
        c = mp.exp(-z)/2
    return c


@mp.extradps(5)
def interval_prob(x1, x2, loc=0, scale=1):
    """
    Compute the probability of x in [x1, x2] for the Laplace distribution.

    Mathematically, this is the same as

        laplace.cdf(x2, loc, scale) - laplace.cdf(x1, loc, scale)

    but when the two CDF values are nearly equal, this function will give
    a more accurate result.

    x1 must be less than or equal to x2.
    """
    loc, scale = _validate_loc_scale(loc, scale)
    x1 = mp.mpf(x1)
    x2 = mp.mpf(x2)
    if x1 > x2:
        raise ValueError('x1 must not be greater than x2')
    if x1 == x2:
        return mp.zero
    z1 = (x1 - loc) / scale
    z2 = (x2 - loc) / scale
    delta = z2 - z1
    if z2 <= 0:
        return -mp.exp(z2)*mp.expm1(-delta)/2
    elif z1 >= 0:
        return -mp.exp(-z1)*mp.expm1(-delta)/2
    else:
        return -(mp.expm1(-z2) + mp.expm1(z1))/2


@mp.extradps(5)
def invcdf(p, loc=0, scale=1):
    """
    Laplace distribution inverse CDF.

    This function is also known as the quantile function or the percent
    point function.
    """
    loc, scale = _validate_loc_scale(loc, scale)
    p = mp.mpf(p)
    if p <= 0.5:
        q = loc + scale * mp.log(2*p)
    else:
        q = loc - scale * mp.log(2 - 2*p)
    return q


@mp.extradps(5)
def invsf(p, loc=0, scale=1):
    """
    Laplace distribution inverse survival function.
    """
    loc, scale = _validate_loc_scale(loc, scale)
    p = mp.mpf(p)
    if p >= 0.5:
        q = loc + scale * mp.log(2 - 2*p)
    else:
        q = loc - scale * mp.log(2*p)
    return q


@mp.extradps(5)
def support(loc=0, scale=1):
    """
    Support of the Laplace distribution.
    """
    loc, scale = _validate_loc_scale(loc, scale)
    return (mp.ninf, mp.inf)


@mp.extradps(5)
def mean(loc=0, scale=1):
    """
    Mean of the Laplace distribution.
    """
    loc, scale = _validate_loc_scale(loc, scale)
    return loc


def mode(loc=0, scale=1):
    """
    Mode of the Laplace distribution.

    The mode of the Laplace distribution is simply `loc`.
    """
    loc, scale = _validate_loc_scale(loc, scale)
    return loc


@mp.extradps(5)
def var(loc=0, scale=1):
    """
    Variance of the Laplace distribution.
    """
    loc, scale = _validate_loc_scale(loc, scale)
    return 2 * scale**2


@mp.extradps(5)
def mle(x, *, loc=None, scale=None):
    """
    Laplace distribution maximum likelihood parameter estimation.

    Returns (loc, scale).

    Note: When len(x) is even, the maximum likelihood estimate for loc
    is not unique, because the likelihood function has a "flat top".
    This function returns the median, with the convention that the
    median of x when len(x) is even is the midpoint of s[n//2-1] and
    s[n//2], where s is sorted(x) and n is len(x).  In fact, *any*
    value between s[n//2 - 1] and s[n//2] is a valid maximum likelihood
    estimate for loc.

    """
    # The estimate of loc is independent of scale, so it doesn't
    # matter if scale is fixed or not.
    if loc is not None:
        # loc is fixed.
        loc_est = mp.mpf(loc)
        x = _seq_to_mp(x)
    else:
        # The MLE for loc is the median of x.
        # When len(x) is even, it is more correct to say that
        # the median is *an* estimate rather than *the* estimate,
        # since the likelihood function does not have a unique
        # maximum in this case.
        x = sorted(_seq_to_mp(x))
        n = len(x)
        m, r = divmod(n, 2)
        if r == 1:
            # n is odd.
            loc_est = x[m]
        else:
            # n is even.
            # The MLE for loc is not unique.  By convention, we use
            # the midpoint between x[m-1] and x[m].
            loc_est = (x[m-1] + x[m])/2

    # At this point, we have loc_est, and x has been converted to a list
    # of mpmath.mpf objects (although we won't need x if scale is fixed).

    if scale is not None:
        # scale is fixed.
        if scale <= 0:
            raise ValueError('scale must be positive.')
        scale_est = mp.mpf(scale)
    else:
        # scale is the mean absolute deviation from loc_est.
        scale_est = _mean([abs(t - loc_est) for t in x])

    return loc_est, scale_est


@mp.extradps(5)
def mom(x):
    """
    Method of moments estimation for the Laplace distribution.

    x must be a sequence of numbers.

    Returns (loc, scale).
    """
    M1 = _mean(x)
    M2 = _mean([mp.mpf(t)**2 for t in x])
    return M1, mp.sqrt((M2 - M1**2)/2)
