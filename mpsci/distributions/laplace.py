"""
Laplace distribution
-------------------

The parameters are mu (the location) and b (the scale).

"""

import mpmath
from mpsci.stats import mean as _mean


__all__ = ['pdf', 'logpdf', 'cdf', 'sf', 'invcdf', 'invsf',
           'mean', 'var', 'mle', 'mom']


def pdf(x, mu=0, b=1):
    """
    Laplace distribution probability density function.
    """
    if b <= 0:
        raise ValueError('b must be positive.')
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        mu = mpmath.mpf(mu)
        b = mpmath.mpf(b)
        return mpmath.exp(-abs(x - mu)/b)/(2*b)


def logpdf(x, mu=0, b=1):
    """
    Log of the PDF of the Laplace distribution.
    """
    if b <= 0:
        raise ValueError('b must be positive.')
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        mu = mpmath.mpf(mu)
        b = mpmath.mpf(b)
        return -abs(x - mu)/b - mpmath.log(2*b)


def cdf(x, mu=0, b=1):
    """
    Laplace distribution cumulative distribution function.
    """
    if b <= 0:
        raise ValueError('b must be positive.')
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        mu = mpmath.mpf(mu)
        b = mpmath.mpf(b)
        z = (x - mu)/b
        if x <= mu:
            c = mpmath.exp(z)/2
        else:
            c = 1 - mpmath.exp(-z)/2
        return c


def sf(x, mu=0, b=1):
    """
    Laplace distribution survival function.
    """
    if b <= 0:
        raise ValueError('b must be positive.')
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        mu = mpmath.mpf(mu)
        b = mpmath.mpf(b)
        z = (x - mu)/b
        if x <= mu:
            c = 1 - mpmath.exp(z)/2
        else:
            c = mpmath.exp(-z)/2
        return c


def invcdf(p, mu=0, b=1):
    """
    Laplace distribution inverse CDF.

    This function is also known as the quantile function or the percent
    point function.
    """
    if b <= 0:
        raise ValueError('b must be positive.')
    with mpmath.extradps(5):
        p = mpmath.mpf(p)
        mu = mpmath.mpf(mu)
        b = mpmath.mpf(b)
        if p <= 0.5:
            q = mu + b*mpmath.log(2*p)
        else:
            q = mu - b*mpmath.log(2 - 2*p)
        return q


def invsf(p, mu=0, b=1):
    """
    Laplace distribution inverse survival function.
    """
    if b <= 0:
        raise ValueError('b must be positive.')
    with mpmath.extradps(5):
        p = mpmath.mpf(p)
        mu = mpmath.mpf(mu)
        b = mpmath.mpf(b)
        if p >= 0.5:
            q = mu + b*mpmath.log(2 - 2*p)
        else:
            q = mu - b*mpmath.log(2*p)
        return q


def mean(mu=0, b=1):
    """
    Mean of the Laplace distribution.
    """
    if b <= 0:
        raise ValueError('b must be positive.')
    with mpmath.extradps(5):
        mu = mpmath.mpf(mu)
        b = mpmath.mpf(b)
        return mu


def var(mu=0, b=1):
    """
    Variance of the Laplace distribution.
    """
    if b <= 0:
        raise ValueError('b must be positive.')
    with mpmath.extradps(5):
        mu = mpmath.mpf(mu)
        b = mpmath.mpf(b)
        return 2*b**2


def mle(x, mu=None, b=None):
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
    with mpmath.extradps(5):
        # The estimate of mu is independent of b, so it doesn't
        # matter if b is fixed or not.
        if mu is not None:
            # mu is fixed.
            mu_est = mpmath.mpf(mu)
            x = list(mpmath.mpf(t) for t in x)
        else:
            # The MLE for mu (the location) is the median of x.
            # When len(x) is even, it is more correct to say that
            # the median is *an* estimate rather than *the* estimate,
            # since the likelihood function does not have a unique
            # maximum in this case.
            x = sorted(mpmath.mpf(t) for t in x)
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
            b_est = mpmath.mpf(b)
        else:
            # b is the mean absolute deviation from mu_est.
            b_est = _mean([abs(t - mu_est) for t in x])

        return mu_est, b_est


def mom(x):
    """
    Method of moments estimation for the Laplace distribution.

    x must be a sequence of numbers.

    Returns (mu, b).
    """
    with mpmath.extradps(5):
        M1 = _mean(x)
        M2 = _mean([mpmath.mpf(t)**2 for t in x])
        return M1, mpmath.sqrt((M2 - M1**2)/2)
