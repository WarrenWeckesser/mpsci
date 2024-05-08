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
from ..fun import digammainv as _digammainv
from ..stats import mean as _mean
from ._common import _validate_p, _validate_x_bounds, _find_bracket, Initial


__all__ = ['pdf', 'logpdf', 'cdf', 'invcdf', 'sf', 'invsf', 'interval_prob',
           'support', 'mean', 'var', 'skewness', 'kurtosis', 'nll', 'mle']


def _validate_k_theta(k, theta):
    if k <= 0:
        raise ValueError('k must be positive')
    if theta <= 0:
        raise ValueError('theta must be positive')
    return mp.mpf(k), mp.mpf(theta)


def pdf(x, k, theta):
    """
    Probability density function for the log-gamma distribution.

    k is the shape parameter of the gamma distribution.
    theta is the scale parameter of the log-gamma distribution.
    """
    with mp.extradps(5):
        x = mp.mpf(x)
        k, theta = _validate_k_theta(k, theta)
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
        k, theta = _validate_k_theta(k, theta)
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
        k, theta = _validate_k_theta(k, theta)
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
        k, theta = _validate_k_theta(k, theta)
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
        k, theta = _validate_k_theta(k, theta)
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
        k, theta = _validate_k_theta(k, theta)
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
        k, theta = _validate_k_theta(k, theta)
        z1 = x1/theta
        z2 = x2/theta
        return mp.gammainc(k, mp.exp(z1), mp.exp(z2), regularized=True)


def support(k, theta):
    """
    Support of the log-gamma distribution.

    k is the shape parameter of the gamma distribution.
    theta is the scale parameter of the log-gamma distribution.
    """
    with mp.extradps(5):
        k, theta = _validate_k_theta(k, theta)
        return (mp.ninf, mp.inf)


def mean(k, theta):
    """
    Mean of the log-gamma distribution.

    k is the shape parameter of the gamma distribution.
    theta is the scale parameter of the log-gamma distribution.
    """
    with mp.extradps(5):
        k, theta = _validate_k_theta(k, theta)
        return theta * mp.psi(0, k)


def var(k, theta):
    """
    Variance of the log-gamma distribution.

    k is the shape parameter of the gamma distribution.
    theta is the scale parameter of the log-gamma distribution.
    """
    with mp.extradps(5):
        k, theta = _validate_k_theta(k, theta)
        return theta**2 * mp.psi(1, k)


def skewness(k, theta):
    """
    Variance of the log-gamma distribution.

    k is the shape parameter of the gamma distribution.
    theta is the scale parameter of the log-gamma distribution.
    """
    with mp.extradps(5):
        k, theta = _validate_k_theta(k, theta)
        return mp.psi(2, k) / mp.psi(1, k)**1.5


def kurtosis(k, theta):
    """
    Excess kurtosis of the log-gamma distribution.

    k is the shape parameter of the gamma distribution.
    theta is the scale parameter of the log-gamma distribution.
    """
    with mp.extradps(5):
        k, theta = _validate_k_theta(k, theta)
        return mp.psi(3, k) / mp.psi(1, k)**2


def nll(x, k, theta):
    """
    Negative log-likelihood for the log-gamma distribution.

    `x` must be a sequence of numbers.
    """
    with mp.extradps(5):
        k, theta = _validate_k_theta(k, theta)
        x = _validate_x_bounds(x, low=mp.ninf, high=mp.inf)
        return -mp.fsum([logpdf(xi, k, theta) for xi in x])


def _mle_shape_scale(x, k0=1, theta0=1):

    def scale_eq(k, scale):
        z = [x1/scale for x1 in x]
        n = len(x)
        return [-n*mp.digamma(k) + mp.fsum(z),
                -n*scale + mp.fsum([x1*(mp.exp(z1) - k)
                                    for x1, z1 in zip(x, z)])]

    x = _validate_x_bounds(x, low=mp.ninf, high=mp.inf)
    k, scale = mp.findroot(scale_eq, [k0, theta0], args=(x,))
    return k, scale


# MLE to do:
# * Handle fixed k.
# * Better default initial guess for the parameters.

def mle(x, k=None, theta=None):
    """
    Maximum likelihood estimation for the log-gamma distribution.

    `x` must be a sequence of numbers.
    """
    if k is not None and not isinstance(k, Initial):
        raise ValueError('Fixed k is not implemented yet.')
    with mp.extradps(5):
        x = _validate_x_bounds(x, low=mp.ninf, high=mp.inf)
        if theta is not None and not isinstance(theta, Initial):
            # theta is fixed, so fit only the shape k.
            _, theta = _validate_k_theta(1, theta)
            xs = [t/theta for t in x]
            # MLE for the shape, assuming the scale is 1:
            k = _digammainv(_mean(xs))
            return k, theta

        # Fit both k and theta.
        k0 = k.initial if isinstance(k, Initial) else 1
        theta0 = theta.initial if isinstance(theta, Initial) else 1
        k0, theta0 = _validate_k_theta(k0, theta0)
        k, theta = _mle_shape_scale(x, k0, theta0)
        return k, theta
