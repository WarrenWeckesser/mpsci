
import mpmath
from ...fun import digammainv


def pdf(x, k, theta):
    """
    Gamma distribution probability density function.

    k is the shape parameter
    theta is the scale parameter (reciprocal of the rate parameter)

    Unlike scipy, a location parameter is not included.
    """
    return mpmath.rgamma(k) / theta**k * x**(k-1) * mpmath.exp(-x/theta)


def cdf(x, k, theta):
    """
    Gamma distribution cumulative distribution function.

    k is the shape parameter
    theta is the scale parameter (reciprocal of the rate parameter)

    Unlike scipy, a location parameter is not included.
    """
    return mpmath.rgamma(k) * mpmath.gammainc(k, 0, x/theta)


def mle(x, k=None, theta=None):
    """
    Gamma distribution maximum likelihood parameter estimation.

    Maximum likelihood estimate for the k (shape) and theta (scale) parameters
    of the gamma distribution.

    x must be a sequence of values.
    """
    meanx = sum(x) / len(x)
    meanlnx = sum(mpmath.log(t) for t in x) / len(x)

    if k is None:
        if theta is None:
            # Solve for k and theta
            s = mpmath.log(meanx) - meanlnx
            k_hat = (3 - s + mpmath.sqrt((s - 3)**2 + 24*s)) / (12*s)
            # XXX This is loop implements a "dumb" convergence criterion.
            # It exits early if the old k equals the new k, but if that never
            # happens, then whatever value k_has after  the last iteration
            # is the value that is returned.
            for _ in range(10):
                oldk = k_hat
                k_hat = k_hat - (mpmath.log(k_hat) - mpmath.psi(0, k_hat) - s)/(1/k_hat - mpmath.psi(1, k_hat))
                if k_hat == oldk:
                    break
            theta_hat = meanx / k_hat
        else:
            # theta is fixed, only solve for k
            k_hat = digammainv(meanlnx - mpmath.log(theta))
            theta_hat = theta
    else:
        if theta is None:
            # Solve for theta, k is fixed.
            k_hat = k
            theta_hat = meanx / k_hat
        else:
            # Both k and theta are fixed.
            k_hat = k
            theta_hat = theta

    if theta is None:
        theta_hat = meanx / k_hat
    else:
        theta_hat = theta

    return k_hat, theta_hat


def nll(x, k, theta):
    """
    Gamma distribution negative log-likelihood.
    """
    N = len(x)
    sumx = sum(x)
    sumlnx = sum(mpmath.log(t) for t in x)

    ll = (k - 1)*sumlnx - sumx/theta - N*k*mpmath.log(theta) - N*mpmath.loggamma(k)
    return -ll


def nll_grad(x, k, theta):
    """
    Gamma distribution gradient of the negative log-likelihood function.
    """
    N = len(x)
    sumx = sum(x)
    sumlnx = sum(mpmath.log(t) for t in x)

    dk = sumlnx - N*mpmath.log(theta) - N*mpmath.digamma(k)
    dtheta = sumx/theta**2 - N*k/theta
    return [-dk, -dtheta]


def nll_hess(x, k, theta):
    """
    Gamma distribution hessian of the negative log-likelihood function.
    """
    N = len(x)
    sumx = sum(x)
    sumlnx = sum(mpmath.log(t) for t in x)

    dk2 = -N*mpmath.psi(1, k)
    dkdtheta = -N/theta
    dtheta2 = -2*sumx/theta**3 + N*k/theta**2

    return mpmath.matrix([[-dk2, -dkdtheta], [-dkdtheta, -dtheta2]])


def nll_invhess(x, k, theta):
    """
    Gamma distribution inverse of the hessian of the negative log-likelihood.
    """
    N = len(x)
    sumx = sum(x)
    sumlnx = sum(mpmath.log(t) for t in x)

    dk2 = -N*mpmath.psi(1, k)
    dkdtheta = -N/theta
    dtheta2 = -2*sumx/theta**3 + N*k/theta**2

    det = dk2*dtheta2 - dkdtheta**2

    return mpmath.matrix([[-dtheta2/det, dkdtheta/det], [dkdtheta/det, -dk2/det]])
