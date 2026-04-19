"""
Laplace distribution
--------------------

The parameters are mu (the location) and b (the scale).

"""

from mpmath import mp
from mpsci.stats import mean as _mean
from ._common import _seq_to_mp, _validate_loc_scale


__all__ = ['pdf', 'logpdf', 'cdf', 'sf', 'invcdf', 'invsf', 'interval_prob',
           'support', 'mean', 'mode', 'var', 'mle', 'mom']


@mp.extradps(5)
def pdf(x, mu=0, b=1):
    """
    Laplace distribution probability density function.
    """
    mu, b = _validate_loc_scale(mu, b, scale_name="b")
    x = mp.mpf(x)
    return mp.exp(-abs(x - mu)/b)/(2*b)


@mp.extradps(5)
def logpdf(x, mu=0, b=1):
    """
    Log of the PDF of the Laplace distribution.
    """
    mu, b = _validate_loc_scale(mu, b, scale_name="b")
    x = mp.mpf(x)
    return -abs(x - mu)/b - mp.log(2*b)


@mp.extradps(5)
def cdf(x, mu=0, b=1):
    """
    Laplace distribution cumulative distribution function.
    """
    mu, b = _validate_loc_scale(mu, b, scale_name="b")
    x = mp.mpf(x)
    z = (x - mu)/b
    if x <= mu:
        c = mp.exp(z)/2
    else:
        c = 1 - mp.exp(-z)/2
    return c


@mp.extradps(5)
def sf(x, mu=0, b=1):
    """
    Laplace distribution survival function.
    """
    mu, b = _validate_loc_scale(mu, b, scale_name="b")
    x = mp.mpf(x)
    z = (x - mu)/b
    if x <= mu:
        c = 1 - mp.exp(z)/2
    else:
        c = mp.exp(-z)/2
    return c


@mp.extradps(5)
def interval_prob(x1, x2, mu=0, b=1):
    """
    Compute the probability of x in [x1, x2] for the Laplace distribution.

    Mathematically, this is the same as

        laplace.cdf(x2, mu, b) - laplace.cdf(x1, mu, b)

    but when the two CDF values are nearly equal, this function will give
    a more accurate result.

    x1 must be less than or equal to x2.
    """
    mu, b = _validate_loc_scale(mu, b, scale_name="b")
    x1 = mp.mpf(x1)
    x2 = mp.mpf(x2)
    if x1 > x2:
        raise ValueError('x1 must not be greater than x2')
    if x1 == x2:
        return mp.zero
    z1 = (x1 - mu)/b
    z2 = (x2 - mu)/b
    delta = z2 - z1
    if z2 <= 0:
        return -mp.exp(z2)*mp.expm1(-delta)/2
    elif z1 >= 0:
        return -mp.exp(-z1)*mp.expm1(-delta)/2
    else:
        return -(mp.expm1(-z2) + mp.expm1(z1))/2


@mp.extradps(5)
def invcdf(p, mu=0, b=1):
    """
    Laplace distribution inverse CDF.

    This function is also known as the quantile function or the percent
    point function.
    """
    mu, b = _validate_loc_scale(mu, b, scale_name="b")
    p = mp.mpf(p)
    if p <= 0.5:
        q = mu + b*mp.log(2*p)
    else:
        q = mu - b*mp.log(2 - 2*p)
    return q


@mp.extradps(5)
def invsf(p, mu=0, b=1):
    """
    Laplace distribution inverse survival function.
    """
    mu, b = _validate_loc_scale(mu, b, scale_name="b")
    p = mp.mpf(p)
    if p >= 0.5:
        q = mu + b*mp.log(2 - 2*p)
    else:
        q = mu - b*mp.log(2*p)
    return q


@mp.extradps(5)
def support(mu=0, b=1):
    """
    Support of the Laplace distribution.
    """
    mu, b = _validate_loc_scale(mu, b, scale_name="b")
    return (mp.ninf, mp.inf)


@mp.extradps(5)
def mean(mu=0, b=1):
    """
    Mean of the Laplace distribution.
    """
    mu, b = _validate_loc_scale(mu, b, scale_name="b")
    return mu


def mode(mu=0, b=1):
    """
    Mode of the Laplace distribution.

    The mode of the Laplace distribution is simply `mu`.
    """
    mu, b = _validate_loc_scale(mu, b, scale_name="b")
    return mu


@mp.extradps(5)
def var(mu=0, b=1):
    """
    Variance of the Laplace distribution.
    """
    mu, b = _validate_loc_scale(mu, b, scale_name="b")
    return 2*b**2


@mp.extradps(5)
def mle(x, *, mu=None, b=None):
    """
    Laplace distribution maximum likelihood parameter estimation.

    Returns (mu, b).

    Note: When len(x) is even, the maximum likelihood estimate for mu
    is not unique, because the likelihood function has a "flat top".
    This function returns the median, with the convention that the
    median of x when len(x) is even is the midpoint of s[n//2-1] and
    s[n//2], where s is sorted(x) and n is len(x).  In fact, *any*
    value between s[n//2 - 1] and s[n//2] is a valid maximum likelihood
    estimate for mu.

    """
    # The estimate of mu is independent of b, so it doesn't
    # matter if b is fixed or not.
    if mu is not None:
        # mu is fixed.
        mu_est = mp.mpf(mu)
        x = _seq_to_mp(x)
    else:
        # The MLE for mu (the location) is the median of x.
        # When len(x) is even, it is more correct to say that
        # the median is *an* estimate rather than *the* estimate,
        # since the likelihood function does not have a unique
        # maximum in this case.
        x = sorted(_seq_to_mp(x))
        n = len(x)
        m, r = divmod(n, 2)
        if r == 1:
            # n is odd.
            mu_est = x[m]
        else:
            # n is even.
            # The MLE for mu is not unique.  By convention, we use
            # the midpoint between x[m-1] and x[m].
            mu_est = (x[m-1] + x[m])/2

    # At this point, we have mu_est, and x has been converted to a list
    # of mpmath.mpf objects (although we won't need x if b is fixed).

    if b is not None:
        # b is fixed.
        if b <= 0:
            raise ValueError('b must be positive.')
        b_est = mp.mpf(b)
    else:
        # b is the mean absolute deviation from mu_est.
        b_est = _mean([abs(t - mu_est) for t in x])

    return mu_est, b_est


@mp.extradps(5)
def mom(x):
    """
    Method of moments estimation for the Laplace distribution.

    x must be a sequence of numbers.

    Returns (mu, b).
    """
    M1 = _mean(x)
    M2 = _mean([mp.mpf(t)**2 for t in x])
    return M1, mp.sqrt((M2 - M1**2)/2)
