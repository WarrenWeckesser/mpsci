"""
Gamma probability distribution
------------------------------

The parameters used here are `k`, the shape parameter, and
`theta`, the scale parameter.

Another common parameterization is shape `k` and the "rate" `lambda`.
`theta` is the reciprocal of `lambda`.

"""

from mpmath import mp
from ._common import _validate_moment_n
from ..fun import digammainv
from . import normal


__all__ = ['pdf', 'logpdf', 'cdf', 'sf', 'interval_prob',
           'mean', 'var', 'skewness', 'kurtosis', 'noncentral_moment',
           'entropy',
           'mom', 'mle',
           'nll', 'nll_grad', 'nll_hess', 'nll_invhess']


def _validate_k_theta(k, theta):
    if k <= 0:
        raise ValueError('k must be positive')
    if theta <= 0:
        raise ValueError('theta must be positive')
    return mp.mpf(k), mp.mpf(theta)


def pdf(x, k, theta):
    """
    Gamma distribution probability density function.

    k is the shape parameter
    theta is the scale parameter (reciprocal of the rate parameter)

    Unlike scipy, a location parameter is not included.
    """
    with mp.extradps(5):
        k, theta = _validate_k_theta(k, theta)
        x = mp.mpf(x)
        if x < 0:
            return mp.zero
        return mp.rgamma(k) / theta**k * x**(k-1) * mp.exp(-x/theta)


def logpdf(x, k, theta):
    """
    Log of the PDF of the gamma distribution.
    """
    with mp.extradps(5):
        k, theta = _validate_k_theta(k, theta)
        x = mp.mpf(x)
        if x < 0:
            return mp.ninf
        return (-mp.loggamma(k) - k*mp.log(theta) +
                (k - 1)*mp.log(x) - x/theta)


def cdf(x, k, theta):
    """
    Gamma distribution cumulative distribution function.

    k is the shape parameter
    theta is the scale parameter (reciprocal of the rate parameter)

    Unlike scipy, a location parameter is not included.
    """
    with mp.extradps(5):
        k, theta = _validate_k_theta(k, theta)
        x = mp.mpf(x)
        if x < 0:
            return mp.zero
        return mp.gammainc(k, 0, x/theta, regularized=True)


def sf(x, k, theta):
    """
    Survival function of the gamma distribution.
    """
    with mp.extradps(5):
        k, theta = _validate_k_theta(k, theta)
        x = mp.mpf(x)
        if x < 0:
            return mp.one
        return mp.gammainc(k, x/theta, mp.inf, regularized=True)


def interval_prob(x1, x2, k, theta):
    """
    Compute the probability of x in [x1, x2] for the gamma distribution.

    Mathematically, this is the same as

        gamma.cdf(x2, k, theta) - gamma.cdf(x1, k, theta)

    but when the two CDF values are nearly equal, this function will give
    a more accurate result.

    x1 must be less than or equal to x2.
    """
    with mp.extradps(5):
        k, theta = _validate_k_theta(k, theta)
        x1 = mp.mpf(x1)
        x2 = mp.mpf(x2)
        if x1 > x2:
            raise ValueError('x1 must not be greater than x2')
        return mp.gammainc(k, x1/theta, x2/theta, regularized=True)


def mean(k, theta):
    """
    Mean of the gamma distribution.
    """
    with mp.extradps(5):
        k, theta = _validate_k_theta(k, theta)
        return k * theta


def var(k, theta):
    """
    Variance of the gamma distribution.
    """
    with mp.extradps(5):
        k, theta = _validate_k_theta(k, theta)
        return k * theta**2


def skewness(k, theta):
    """
    Skewness of the gamma distribution.
    """
    with mp.extradps(5):
        k, theta = _validate_k_theta(k, theta)
        return 2/mp.sqrt(k)


def kurtosis(k, theta):
    """
    Excess kurtosis of the gamma distribution.
    """
    with mp.extradps(5):
        k, theta = _validate_k_theta(k, theta)
        return 6/k


def noncentral_moment(n, k, theta):
    """
    n-th noncentral moment of the gamma distribution.

    n must be a nonnegative integer.
    """
    n = _validate_moment_n(n)
    with mp.extradps(5):
        k, theta = _validate_k_theta(k, theta)
        if n == 0:
            return mp.one
        return theta**n * mp.gammaprod([k + n], [k])


def entropy(k, theta):
    """
    Differential entropy of the gamma distribution.
    """
    with mp.extradps(5):
        k, theta = _validate_k_theta(k, theta)
        return k + mp.log(theta) + mp.loggamma(k) + (1 - k)*mp.psi(0, k)


def mom(x):
    """
    Parameter estimation by the method of moments for the gamma distribution.

    x must be a sequence of values.

    Returns the estimates of the shape k and the scale theta.
    """
    n = len(x)
    with mp.extradps(5):
        m1 = mp.fsum(x) / n
        m2 = mp.fsum([mp.power(t, 2) for t in x]) / n
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
    meanx = mp.fsum(x) / len(x)
    meanlnx = mp.fsum(mp.log(t) for t in x) / len(x)

    if k is None:
        if theta is None:
            # Solve for k and theta
            s = mp.log(meanx) - meanlnx
            k_hat = (3 - s + mp.sqrt((s - 3)**2 + 24*s)) / (12*s)
            # XXX This loop implements a "dumb" convergence criterion.
            # It exits early if the old k equals the new k, but if that never
            # happens, then whatever value k_hat has after  the last iteration
            # is the value that is returned.
            for _ in range(10):
                oldk = k_hat
                delta = ((mp.log(k_hat) - mp.psi(0, k_hat) - s) /
                         (1/k_hat - mp.psi(1, k_hat)))
                k_hat = k_hat - delta
                if k_hat == oldk:
                    break
            theta_hat = meanx / k_hat
        else:
            # theta is fixed, only solve for k
            theta = mp.mpf(theta)
            k_hat = digammainv(meanlnx - mp.log(theta))
            theta_hat = theta
    else:
        if theta is None:
            # Solve for theta, k is fixed.
            k_hat = mp.mpf(k)
            theta_hat = meanx / k_hat
        else:
            # Both k and theta are fixed.
            k_hat = mp.mpf(k)
            theta_hat = mp.mpf(theta)

    return k_hat, theta_hat


def nll(x, k, theta):
    """
    Gamma distribution negative log-likelihood.
    """
    with mp.extradps(5):
        k, theta = _validate_k_theta(k, theta)

        N = len(x)
        sumx = mp.fsum(x)
        sumlnx = mp.fsum(mp.log(t) for t in x)

        ll = ((k - 1)*sumlnx - sumx/theta - N*k*mp.log(theta) -
              N*mp.loggamma(k))
        return -ll


def nll_grad(x, k, theta):
    """
    Gamma distribution gradient of the negative log-likelihood function.
    """
    with mp.extradps(5):
        k, theta = _validate_k_theta(k, theta)

        N = len(x)
        sumx = mp.fsum(x)
        sumlnx = mp.fsum(mp.log(t) for t in x)

        dk = sumlnx - N*mp.log(theta) - N*mp.digamma(k)
        dtheta = sumx/theta**2 - N*k/theta
        return [-dk, -dtheta]


def nll_hess(x, k, theta):
    """
    Gamma distribution hessian of the negative log-likelihood function.
    """
    with mp.extradps(5):
        k, theta = _validate_k_theta(k, theta)

        N = len(x)
        sumx = mp.fsum(x)
        # sumlnx = mp.fsum(mpmath.log(t) for t in x)

        dk2 = -N*mp.psi(1, k)
        dkdtheta = -N/theta
        dtheta2 = -2*sumx/theta**3 + N*k/theta**2

        return mp.matrix([[-dk2, -dkdtheta], [-dkdtheta, -dtheta2]])


def nll_invhess(x, k, theta):
    """
    Gamma distribution inverse of the hessian of the negative log-likelihood.
    """
    with mp.extradps(5):
        k, theta = _validate_k_theta(k, theta)

        N = len(x)
        sumx = mp.fsum(x)
        # sumlnx = mp.fsum(mpmath.log(t) for t in x)

        dk2 = -N*mp.psi(1, k)
        dkdtheta = -N/theta
        dtheta2 = -2*sumx/theta**3 + N*k/theta**2

        det = dk2*dtheta2 - dkdtheta**2

        return mp.matrix([[-dtheta2/det, dkdtheta/det],
                          [dkdtheta/det, -dk2/det]])


#
# The following are experimental and not thoroughly checked.
#

def _mle_se(x, k, theta):
    """
    Standard errors of the MLE estimates k and theta for the sequence x.

    This function assumes that both k and theta were estimated from x.
    """
    with mp.extradps(5):
        k, theta = _validate_k_theta(k, theta)
        invhess_diag = _nll_invhess_diag(x, k, theta)
        return (mp.sqrt(invhess_diag[0]), mp.sqrt(invhess_diag[1]))


def _mle_ci(x, k, theta, alpha):
    """
    Confidence intervals of the MLE estimates k and theta for the sequence x.

    This function assumes that both k and theta were estimated from x.

    The values are based on the asymptotic normality of the estimates.
    These are not the exact confidence intervals.
    """
    with mp.extradps(5):
        k, theta = _validate_k_theta(k, theta)
        kse, thetase = _mle_se(x, k, theta)
        w = normal.invcdf(alpha/2)
        return (k + w*kse, k - w*kse), (theta + w*thetase, theta - w*thetase)


def _nll_invhess_diag(x, k, theta):
    """
    Diagonal elements of the inverse of the hessian of the negative
    log-likelihood of the gamma distribution.
    """
    with mp.extradps(5):
        k, theta = _validate_k_theta(k, theta)

        N = len(x)
        sumx = mp.fsum(x)
        # sumlnx = mp.fsum(mp.log(t) for t in x)

        dk2 = -N*mp.psi(1, k)
        dkdtheta = -N/theta
        dtheta2 = -2*sumx/theta**3 + N*k/theta**2

        det = dk2*dtheta2 - dkdtheta**2

        return (-dtheta2/det, -dk2/det)
