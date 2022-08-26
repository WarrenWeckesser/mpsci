"""
Log-gamma probability distribution
----------------------------------

* k is the shape parameter of the gamma distribution.
* theta is the scale parameter of the log-gamma distribution.

In SciPy, this distribution is implemented as `scipy.stats.loggamma`.

In the Wolfram language, this distribution is called `ExpGammaDistribution`.

Unlike SciPy and Wolfram, a location parameter is not included in this
implementation of the log-gamma distribution.

"""

import mpmath
from ._common import _validate_p


__all__ = ['pdf', 'logpdf', 'cdf', 'invcdf', 'sf', 'invsf', 'interval_prob',
           'mean', 'var', 'skewness', 'kurtosis']


def pdf(x, k, theta):
    """
    Probability density function for the log-gamma distribution.

    k is the shape parameter of the gamma distribution.
    theta is the scale parameter of the log-gamma distribution.
    """
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        k = mpmath.mpf(k)
        theta = mpmath.mpf(theta)
        z = x/theta
        return mpmath.exp(k*z - mpmath.exp(z))/mpmath.gamma(k)/theta


def logpdf(x, k, theta):
    """
    Log of the PDF of the log-gamma distribution.

    k is the shape parameter of the gamma distribution.
    theta is the scale parameter of the log-gamma distribution.
    """
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        k = mpmath.mpf(k)
        theta = mpmath.mpf(theta)
        z = x/theta
        return (k*z - mpmath.exp(z)) - mpmath.loggamma(k) - mpmath.log(theta)


def cdf(x, k, theta):
    """
    Cumulative distribution function of the log-gamma distribution.

    k is the shape parameter of the gamma distribution.
    theta is the scale parameter of the log-gamma distribution.
    """
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        k = mpmath.mpf(k)
        theta = mpmath.mpf(theta)
        z = x/theta
        return mpmath.gammainc(k, 0, mpmath.exp(z), regularized=True)


def invcdf(p, k, theta, x0):
    """
    Inverse of the CDF for the log-gamma distribution.

    Also known as the quantile function.

    k is the shape parameter of the gamma distribution.
    theta is the scale parameter of the log-gamma distribution.
    x0 is an initial guess for the quantile.
    """
    with mpmath.extradps(5):
        p = _validate_p(p)
        if p == 0:
            return mpmath.ninf
        if p == 1:
            return mpmath.inf
        k = mpmath.mpf(k)
        theta = mpmath.mpf(theta)
        root = mpmath.findroot(lambda t: cdf(t, k, theta) - p, x0)
        return root


def sf(x, k, theta):
    """
    Survival function of the log-gamma distribution.

    k is the shape parameter of the gamma distribution.
    theta is the scale parameter of the log-gamma distribution.
    """
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        k = mpmath.mpf(k)
        theta = mpmath.mpf(theta)
        z = x/theta
        return mpmath.gammainc(k, mpmath.exp(z), mpmath.inf, regularized=True)


def invsf(p, k, theta, x0):
    """
    Inverse of the survival functin for the log-gamma distribution.

    k is the shape parameter of the gamma distribution.
    theta is the scale parameter of the log-gamma distribution.
    x0 is an initial guess for the quantile.
    """
    with mpmath.extradps(5):
        p = _validate_p(p)
        if p == 0:
            return mpmath.inf
        if p == 1:
            return mpmath.ninf
        k = mpmath.mpf(k)
        theta = mpmath.mpf(theta)
        root = mpmath.findroot(lambda t: sf(t, k, theta) - p, x0)
        return root


def interval_prob(x1, x2, k, theta):
    """
    Compute the probability of x in [x1, x2] for the log-gamma distribution.

    Mathematically, this is the same as

        loggamma.cdf(x2, k, theta) - loggamma.cdf(x1, k, theta)

    but when the two CDF values are nearly equal, this function will give
    a more accurate result.

    x1 must be less than or equal to x2.

    k is the shape parameter of the gamma distribution.
    theta is the scale parameter of the log-gamma distribution.
    """
    if x1 > x2:
        raise ValueError('x1 must not be greater than x2')

    with mpmath.extradps(5):
        x1 = mpmath.mpf(x1)
        x2 = mpmath.mpf(x2)
        k = mpmath.mpf(k)
        theta = mpmath.mpf(theta)
        z1 = x1/theta
        z2 = x2/theta
        return mpmath.gammainc(k, mpmath.exp(z1), mpmath.exp(z2),
                               regularized=True)


def mean(k, theta):
    """
    Mean of the log-gamma distribution.

    k is the shape parameter of the gamma distribution.
    theta is the scale parameter of the log-gamma distribution.
    """
    with mpmath.extradps(5):
        k = mpmath.mpf(k)
        theta = mpmath.mpf(theta)
        return theta * mpmath.psi(0, k)


def var(k, theta):
    """
    Variance of the log-gamma distribution.

    k is the shape parameter of the gamma distribution.
    theta is the scale parameter of the log-gamma distribution.
    """
    with mpmath.extradps(5):
        k = mpmath.mpf(k)
        theta = mpmath.mpf(theta)
        return theta**2 * mpmath.psi(1, k)


def skewness(k, theta):
    """
    Variance of the log-gamma distribution.

    k is the shape parameter of the gamma distribution.
    theta is the scale parameter of the log-gamma distribution.
    """
    with mpmath.extradps(5):
        k = mpmath.mpf(k)
        theta = mpmath.mpf(theta)
        return mpmath.psi(2, k) / mpmath.psi(1, k)**1.5


def kurtosis(k, theta):
    """
    Excess kurtosis of the log-gamma distribution.

    k is the shape parameter of the gamma distribution.
    theta is the scale parameter of the log-gamma distribution.
    """
    with mpmath.extradps(5):
        k = mpmath.mpf(k)
        theta = mpmath.mpf(theta)
        return mpmath.psi(3, k) / mpmath.psi(1, k)**2
