"""
Gamma probability distribution
------------------------------
"""

import mpmath
from ..fun import digammainv


__all__ = ['pdf', 'logpdf', 'cdf', 'sf', 'interval_prob',
           'mean', 'var', 'mom', 'mle',
           'nll', 'nll_grad', 'nll_hess', 'nll_invhess']


def pdf(x, k, theta):
    """
    Gamma distribution probability density function.

    k is the shape parameter
    theta is the scale parameter (reciprocal of the rate parameter)

    Unlike scipy, a location parameter is not included.
    """
    x = mpmath.mpf(x)
    k = mpmath.mpf(k)
    theta = mpmath.mpf(theta)
    return mpmath.rgamma(k) / theta**k * x**(k-1) * mpmath.exp(-x/theta)


def logpdf(x, k, theta):
    """
    Log of the PDF of the gamma distribution.
    """
    x = mpmath.mpf(x)
    k = mpmath.mpf(k)
    theta = mpmath.mpf(theta)
    return (-mpmath.loggamma(k) - k*mpmath.log(theta) +
            (k - 1)*mpmath.log(x) - x/theta)


def cdf(x, k, theta):
    """
    Gamma distribution cumulative distribution function.

    k is the shape parameter
    theta is the scale parameter (reciprocal of the rate parameter)

    Unlike scipy, a location parameter is not included.
    """
    x = mpmath.mpf(x)
    k = mpmath.mpf(k)
    theta = mpmath.mpf(theta)
    return mpmath.gammainc(k, 0, x/theta, regularized=True)


def sf(x, k, theta):
    """
    Survival function of the gamma distribution.
    """
    x = mpmath.mpf(x)
    k = mpmath.mpf(k)
    theta = mpmath.mpf(theta)
    return mpmath.gammainc(k, x/theta, mpmath.inf, regularized=True)


def interval_prob(x1, x2, k, theta):
    """
    Compute the probability of x in [x1, x2] for the gamma distribution.

    Mathematically, this is the same as

        gamma.cdf(x2, k, theta) - gamma.cdf(x1, k, theta).

    but when the two CDF values are nearly equal, this function will give
    a more accurate result.

    x1 must be less than or equal to x2.
    """
    if x1 > x2:
        raise ValueError('x1 must not be greater than x2')

    with mpmath.extradps(5):
        x1 = mpmath.mpf(x1)
        x2 = mpmath.mpf(x2)
        k = mpmath.mpf(k)
        theta = mpmath.mpf(theta)
        return mpmath.gammainc(k, x1/theta, x2/theta, regularized=True)


def mean(k, theta):
    """
    Mean of the gamma distribution.
    """
    k = mpmath.mpf(k)
    theta = mpmath.mpf(theta)
    return k * theta


def var(k, theta):
    """
    Variance of the gamma distribution.
    """
    k = mpmath.mpf(k)
    theta = mpmath.mpf(theta)
    return k * theta**2


def mom(x):
    """
    Parameter estimation by the method of moments for the gamma distribution.

    x must be a sequence of values.

    Returns the estimates of the shape k and the scale theta.
    """
    n = len(x)
    with mpmath.extradps(5):
        m1 = mpmath.fsum(x) / n
        m2 = mpmath.fsum([mpmath.power(t, 2) for t in x]) / n
        m1sq = m1**2
        k = m1sq / (m2 - m1sq)
        theta = (m2 - m1sq) / m1
    return k, theta


def mle(x, k=None, theta=None):
    """
    Gamma distribution maximum likelihood parameter estimation.

    Maximum likelihood estimate for the k (shape) and theta (scale) parameters
    of the gamma distribution.

    x must be a sequence of values.
    """
    meanx = mpmath.fsum(x) / len(x)
    meanlnx = mpmath.fsum(mpmath.log(t) for t in x) / len(x)

    if k is None:
        if theta is None:
            # Solve for k and theta
            s = mpmath.log(meanx) - meanlnx
            k_hat = (3 - s + mpmath.sqrt((s - 3)**2 + 24*s)) / (12*s)
            # XXX This loop implements a "dumb" convergence criterion.
            # It exits early if the old k equals the new k, but if that never
            # happens, then whatever value k_hat has after  the last iteration
            # is the value that is returned.
            for _ in range(10):
                oldk = k_hat
                delta = ((mpmath.log(k_hat) - mpmath.psi(0, k_hat) - s) /
                         (1/k_hat - mpmath.psi(1, k_hat)))
                k_hat = k_hat - delta
                if k_hat == oldk:
                    break
            theta_hat = meanx / k_hat
        else:
            # theta is fixed, only solve for k
            theta = mpmath.mpf(theta)
            k_hat = digammainv(meanlnx - mpmath.log(theta))
            theta_hat = theta
    else:
        if theta is None:
            # Solve for theta, k is fixed.
            k_hat = mpmath.mpf(k)
            theta_hat = meanx / k_hat
        else:
            # Both k and theta are fixed.
            k_hat = mpmath.mpf(k)
            theta_hat = mpmath.mpf(theta)

    return k_hat, theta_hat


def nll(x, k, theta):
    """
    Gamma distribution negative log-likelihood.
    """
    k = mpmath.mpf(k)
    theta = mpmath.mpf(theta)

    N = len(x)
    sumx = mpmath.fsum(x)
    sumlnx = mpmath.fsum(mpmath.log(t) for t in x)

    ll = ((k - 1)*sumlnx - sumx/theta - N*k*mpmath.log(theta) -
          N*mpmath.loggamma(k))
    return -ll


def nll_grad(x, k, theta):
    """
    Gamma distribution gradient of the negative log-likelihood function.
    """
    k = mpmath.mpf(k)
    theta = mpmath.mpf(theta)

    N = len(x)
    sumx = mpmath.fsum(x)
    sumlnx = mpmath.fsum(mpmath.log(t) for t in x)

    dk = sumlnx - N*mpmath.log(theta) - N*mpmath.digamma(k)
    dtheta = sumx/theta**2 - N*k/theta
    return [-dk, -dtheta]


def nll_hess(x, k, theta):
    """
    Gamma distribution hessian of the negative log-likelihood function.
    """
    k = mpmath.mpf(k)
    theta = mpmath.mpf(theta)

    N = len(x)
    sumx = mpmath.fsum(x)
    # sumlnx = mpmath.fsum(mpmath.log(t) for t in x)

    dk2 = -N*mpmath.psi(1, k)
    dkdtheta = -N/theta
    dtheta2 = -2*sumx/theta**3 + N*k/theta**2

    return mpmath.matrix([[-dk2, -dkdtheta], [-dkdtheta, -dtheta2]])


def nll_invhess(x, k, theta):
    """
    Gamma distribution inverse of the hessian of the negative log-likelihood.
    """
    k = mpmath.mpf(k)
    theta = mpmath.mpf(theta)

    N = len(x)
    sumx = mpmath.fsum(x)
    # sumlnx = mpmath.fsum(mpmath.log(t) for t in x)

    dk2 = -N*mpmath.psi(1, k)
    dkdtheta = -N/theta
    dtheta2 = -2*sumx/theta**3 + N*k/theta**2

    det = dk2*dtheta2 - dkdtheta**2

    return mpmath.matrix([[-dtheta2/det, dkdtheta/det],
                          [dkdtheta/det, -dk2/det]])
