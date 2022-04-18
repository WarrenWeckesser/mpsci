"""
Log-normal distribution
-----------------------

The parameters mu and sigma are the mean and standard deviation
of the underlying normal distribution.  These are not the same
parameters as used in `scipy.stats.lognorm`.
"""

import mpmath


__all__ = ['pdf', 'logpdf', 'cdf', 'sf', 'invcdf', 'invsf',
           'mean', 'var', 'skewness', 'kurtosis', 'mle', 'mom']


def _validate_sigma(sigma):
    if sigma <= 0:
        raise ValueError('sigma must be positive')


def pdf(x, mu=0, sigma=1):
    """
    Log-normal distribution probability density function.
    """
    _validate_sigma(sigma)
    if x <= 0:
        return mpmath.mp.zero
    x = mpmath.mpf(x)
    lnx = mpmath.log(x)
    return mpmath.npdf(lnx, mu, sigma) / x


def logpdf(x, mu=0, sigma=1):
    """
    Natural logarithm of the PDF of the log-normal distribution.
    """
    _validate_sigma(sigma)
    if x <= 0:
        return -mpmath.inf
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        mu = mpmath.mpf(mu)
        sigma = mpmath.mpf(sigma)
        lnx = mpmath.log(x)
        t = -(lnx - mu)**2 / (2*sigma**2)
        return (-mpmath.log(x) - mpmath.log(sigma)
                - 0.5*mpmath.log(2*mpmath.pi) + t)


def cdf(x, mu=0, sigma=1):
    """
    Log-normal distribution cumulative distribution function.
    """
    _validate_sigma(sigma)
    if x <= 0:
        return mpmath.mp.zero
    lnx = mpmath.log(x)
    return mpmath.ncdf(lnx, mu, sigma)


def sf(x, mu=0, sigma=1):
    """
    Log-normal distribution survival function.
    """
    _validate_sigma(sigma)
    if x <= 0:
        return mpmath.mp.one
    lnx = mpmath.log(x)
    return mpmath.ncdf(-lnx, -mu, sigma)


def invcdf(p, mu=0, sigma=1):
    """
    Log-normal distribution inverse CDF.

    This function is also known as the quantile function or the percent
    point function.
    """
    _validate_sigma(sigma)
    if p < 0 or p > 1:
        return mpmath.nan

    with mpmath.extradps(5):
        p = mpmath.mpf(p)
        mu = mpmath.mpf(mu)
        sigma = mpmath.mpf(sigma)
        a = mpmath.erfinv(2*p - 1)
        x = mpmath.exp(mpmath.sqrt(2)*sigma*a + mu)

    return x


def invsf(p, mu=0, sigma=1):
    """
    Log-normal distribution inverse survival function.
    """
    _validate_sigma(sigma)
    return invcdf(1 - p, mu, sigma)


def mean(mu=0, sigma=1):
    """
    Mean of the lognormal distribution.
    """
    _validate_sigma(sigma)
    mu = mpmath.mpf(mu)
    sigma = mpmath.mpf(sigma)
    return mpmath.exp(mu + sigma**2/2)


def var(mu=0, sigma=1):
    """
    Variance of the lognormal distribution.
    """
    _validate_sigma(sigma)
    mu = mpmath.mpf(mu)
    sigma = mpmath.mpf(sigma)
    sigma2 = sigma**2
    return mpmath.expm1(sigma2) * mpmath.exp(2*mu + sigma2)


def skewness(mu=0, sigma=1):
    """
    Skewness of the lognormal distribution.
    """
    _validate_sigma(sigma)
    sigma2 = sigma**2
    return (mpmath.exp(sigma2) + 2) * mpmath.sqrt(mpmath.expm1(sigma2))


def kurtosis(mu=0, sigma=1):
    """
    Kurtosis of the lognormal distribution.
    """
    _validate_sigma(sigma)
    sigma2 = sigma**2
    return (mpmath.exp(4*sigma2) + 2*mpmath.exp(3*sigma2)
            + 3*mpmath.exp(2*sigma2) - 6)


# XXX Add standard errors and confidence intervals for the fitted parameters.


def _validate_x(x):
    if any(t <= 0 for t in x):
        raise ValueError('All values in x must be greater than 0.')


def mle(x):
    """
    Log-normal distribution maximum likelihood parameter estimation.

    x must be a sequence of numbers.

    Returns (mu, sigma).
    """
    _validate_x(x)
    lnx = [mpmath.log(t) for t in x]
    N = len(x)
    meanx = sum(lnx) / N
    var = sum((lnxi - meanx)**2 for lnxi in lnx) / N
    sigma = mpmath.sqrt(var)
    return meanx, sigma


def mom(x):
    """
    Method of moments parameter estimation for the log-normal distribution.

    x must be a sequence of numbers.

    Returns (mu, sigma).
    """
    _validate_x(x)
    with mpmath.extradps(5):
        logsumx = mpmath.log(mpmath.fsum(x))
        logsumx2 = mpmath.log(mpmath.fsum([mpmath.mpf(t)**2 for t in x]))
        logn = mpmath.log(len(x))
        mu = -logsumx2/2 + 2*logsumx - 3*logn/2
        sigma = mpmath.sqrt(logsumx2 - 2*logsumx + logn)
        return mu, sigma
