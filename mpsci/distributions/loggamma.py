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

from mpmath import mp
from ._common import _validate_p, _find_bracket


__all__ = ['pdf', 'logpdf', 'cdf', 'invcdf', 'sf', 'invsf', 'interval_prob',
           'mean', 'var', 'skewness', 'kurtosis']


def pdf(x, k, theta):
    """
    Probability density function for the log-gamma distribution.

    k is the shape parameter of the gamma distribution.
    theta is the scale parameter of the log-gamma distribution.
    """
    with mp.extradps(5):
        x = mp.mpf(x)
        k = mp.mpf(k)
        theta = mp.mpf(theta)
        z = x/theta
        return mp.exp(k*z - mp.exp(z))/mp.gamma(k)/theta


def logpdf(x, k, theta):
    """
    Log of the PDF of the log-gamma distribution.

    k is the shape parameter of the gamma distribution.
    theta is the scale parameter of the log-gamma distribution.
    """
    with mp.extradps(5):
        x = mp.mpf(x)
        k = mp.mpf(k)
        theta = mp.mpf(theta)
        z = x/theta
        return (k*z - mp.exp(z)) - mp.loggamma(k) - mp.log(theta)


def cdf(x, k, theta):
    """
    Cumulative distribution function of the log-gamma distribution.

    k is the shape parameter of the gamma distribution.
    theta is the scale parameter of the log-gamma distribution.
    """
    with mp.extradps(5):
        x = mp.mpf(x)
        k = mp.mpf(k)
        theta = mp.mpf(theta)
        z = x/theta
        return mp.gammainc(k, 0, mp.exp(z), regularized=True)


def invcdf(p, k, theta):
    """
    Inverse of the CDF for the log-gamma distribution.

    Also known as the quantile function.

    k is the shape parameter of the gamma distribution.
    theta is the scale parameter of the log-gamma distribution.
    """
    with mp.extradps(5):
        p = _validate_p(p)
        if p == 0:
            return mp.ninf
        if p == 1:
            return mp.inf
        k = mp.mpf(k)
        theta = mp.mpf(theta)
        x0, x1 = _find_bracket(lambda t: cdf(t, k, theta), p, -mp.inf, mp.inf)
        root = mp.findroot(lambda t: cdf(t, k, theta) - p, x0=(x0, x1))
        return root


def sf(x, k, theta):
    """
    Survival function of the log-gamma distribution.

    k is the shape parameter of the gamma distribution.
    theta is the scale parameter of the log-gamma distribution.
    """
    with mp.extradps(5):
        x = mp.mpf(x)
        k = mp.mpf(k)
        theta = mp.mpf(theta)
        z = x/theta
        return mp.gammainc(k, mp.exp(z), mp.inf, regularized=True)


def invsf(p, k, theta):
    """
    Inverse of the survival functin for the log-gamma distribution.

    k is the shape parameter of the gamma distribution.
    theta is the scale parameter of the log-gamma distribution.
    """
    with mp.extradps(5):
        p = _validate_p(p)
        if p == 0:
            return mp.inf
        if p == 1:
            return mp.ninf
        k = mp.mpf(k)
        theta = mp.mpf(theta)
        x0, x1 = _find_bracket(lambda t: sf(t, k, theta), p, -mp.inf, mp.inf)
        root = mp.findroot(lambda t: sf(t, k, theta) - p, x0)
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

    with mp.extradps(5):
        x1 = mp.mpf(x1)
        x2 = mp.mpf(x2)
        k = mp.mpf(k)
        theta = mp.mpf(theta)
        z1 = x1/theta
        z2 = x2/theta
        return mp.gammainc(k, mp.exp(z1), mp.exp(z2), regularized=True)


def mean(k, theta):
    """
    Mean of the log-gamma distribution.

    k is the shape parameter of the gamma distribution.
    theta is the scale parameter of the log-gamma distribution.
    """
    with mp.extradps(5):
        k = mp.mpf(k)
        theta = mp.mpf(theta)
        return theta * mp.psi(0, k)


def var(k, theta):
    """
    Variance of the log-gamma distribution.

    k is the shape parameter of the gamma distribution.
    theta is the scale parameter of the log-gamma distribution.
    """
    with mp.extradps(5):
        k = mp.mpf(k)
        theta = mp.mpf(theta)
        return theta**2 * mp.psi(1, k)


def skewness(k, theta):
    """
    Variance of the log-gamma distribution.

    k is the shape parameter of the gamma distribution.
    theta is the scale parameter of the log-gamma distribution.
    """
    with mp.extradps(5):
        k = mp.mpf(k)
        theta = mp.mpf(theta)
        return mp.psi(2, k) / mp.psi(1, k)**1.5


def kurtosis(k, theta):
    """
    Excess kurtosis of the log-gamma distribution.

    k is the shape parameter of the gamma distribution.
    theta is the scale parameter of the log-gamma distribution.
    """
    with mp.extradps(5):
        k = mp.mpf(k)
        theta = mp.mpf(theta)
        return mp.psi(3, k) / mp.psi(1, k)**2
