"""
Log-normal distribution
-----------------------

The parameters mu and sigma are the mean and standard deviation
of the underlying normal distribution.  These are not the same
parameters as used in `scipy.stats.lognorm`.
"""

import operator
from mpmath import mp
from ._common import _validate_p


__all__ = ['pdf', 'logpdf', 'cdf', 'sf', 'invcdf', 'invsf',
           'mean', 'var', 'skewness', 'kurtosis',
           'noncentral_moment',
           'mle', 'mom']


def _validate_sigma(sigma):
    if sigma <= 0:
        raise ValueError('sigma must be positive')
    return mp.mpf(sigma)


def pdf(x, mu=0, sigma=1):
    """
    Log-normal distribution probability density function.
    """
    with mp.extradps(5):
        sigma = _validate_sigma(sigma)
        x = mp.mpf(x)
        if x <= 0:
            return mp.zero
        lnx = mp.log(x)
        return mp.npdf(lnx, mu, sigma) / x


def logpdf(x, mu=0, sigma=1):
    """
    Natural logarithm of the PDF of the log-normal distribution.
    """
    with mp.extradps(5):
        sigma = _validate_sigma(sigma)
        mu = mp.mpf(mu)
        x = mp.mpf(x)
        if x <= 0:
            return -mp.inf
        lnx = mp.log(x)
        t = -(lnx - mu)**2 / (2*sigma**2)
        return (-mp.log(x) - mp.log(sigma)
                - 0.5*mp.log(2*mp.pi) + t)


def cdf(x, mu=0, sigma=1):
    """
    Log-normal distribution cumulative distribution function.
    """
    with mp.extradps(5):
        sigma = _validate_sigma(sigma)
        mu = mp.mpf(mu)
        x = mp.mpf(x)
        if x <= 0:
            return mp.zero
        lnx = mp.log(x)
        return mp.ncdf(lnx, mu, sigma)


def sf(x, mu=0, sigma=1):
    """
    Log-normal distribution survival function.
    """
    with mp.extradps(5):
        sigma = _validate_sigma(sigma)
        mu = mp.mpf(mu)
        x = mp.mpf(x)
        if x <= 0:
            return mp.mp.one
        lnx = mp.log(x)
        return mp.ncdf(-lnx, -mu, sigma)


def invcdf(p, mu=0, sigma=1):
    """
    Log-normal distribution inverse CDF.

    This function is also known as the quantile function or the percent
    point function.
    """
    with mp.extradps(5):
        p = _validate_p(p)
        sigma = _validate_sigma(sigma)
        mu = mp.mpf(mu)
        a = mp.erfinv(2*p - 1)
        x = mp.exp(mp.sqrt(2)*sigma*a + mu)
        return x


def invsf(p, mu=0, sigma=1):
    """
    Log-normal distribution inverse survival function.
    """
    with mp.extradps(5):
        p = _validate_p(p)
        sigma = _validate_sigma(sigma)
        mu = mp.mpf(mu)
        return invcdf(1 - p, mu, sigma)


def mean(mu=0, sigma=1):
    """
    Mean of the lognormal distribution.
    """
    with mp.extradps(5):
        sigma = _validate_sigma(sigma)
        mu = mp.mpf(mu)
        return mp.exp(mu + sigma**2/2)


def var(mu=0, sigma=1):
    """
    Variance of the lognormal distribution.
    """
    with mp.extradps(5):
        sigma = _validate_sigma(sigma)
        mu = mp.mpf(mu)
        sigma2 = sigma**2
        return mp.expm1(sigma2) * mp.exp(2*mu + sigma2)


def skewness(mu=0, sigma=1):
    """
    Skewness of the lognormal distribution.
    """
    with mp.extradps(5):
        sigma = _validate_sigma(sigma)
        sigma2 = sigma**2
        return (mp.exp(sigma2) + 2) * mp.sqrt(mp.expm1(sigma2))


def kurtosis(mu=0, sigma=1):
    """
    Kurtosis of the lognormal distribution.
    """
    with mp.extradps(5):
        sigma = _validate_sigma(sigma)
        sigma2 = sigma**2
        return (mp.exp(4*sigma2) + 2*mp.exp(3*sigma2)
                + 3*mp.exp(2*sigma2) - 6)


def noncentral_moment(n, mu=0, sigma=1):
    """
    Noncentral moment of the lognormal distribution.
    """
    try:
        n = operator.index(n)
    except TypeError:
        raise TypeError('n must be an integer')
    if n < 0:
        raise ValueError('n must be nonnegative')

    with mp.extradps(5):
        sigma = _validate_sigma(sigma)
        mu = mp.mpf(mu)
        return mp.exp(n*mu + n**2*sigma**2/2)


# XXX Add standard errors and confidence intervals for the fitted parameters.


def _validate_x(x):
    if any(t <= 0 for t in x):
        raise ValueError('All values in x must be greater than 0.')
    return [mp.mpf(t) for t in x]


def mle(x):
    """
    Log-normal distribution maximum likelihood parameter estimation.

    x must be a sequence of numbers.

    Returns (mu, sigma).
    """
    with mp.extradps(5):
        x = _validate_x(x)
        lnx = [mp.log(t) for t in x]
        N = len(x)
        meanx = sum(lnx) / N
        var = sum((lnxi - meanx)**2 for lnxi in lnx) / N
        sigma = mp.sqrt(var)
        return meanx, sigma


def mom(x):
    """
    Method of moments parameter estimation for the log-normal distribution.

    x must be a sequence of numbers.

    Returns (mu, sigma).
    """
    with mp.extradps(5):
        x = _validate_x(x)
        logsumx = mp.log(mp.fsum(x))
        logsumx2 = mp.log(mp.fsum([mp.mpf(t)**2 for t in x]))
        logn = mp.log(len(x))
        mu = -logsumx2/2 + 2*logsumx - 3*logn/2
        sigma = mp.sqrt(logsumx2 - 2*logsumx + logn)
        return mu, sigma
