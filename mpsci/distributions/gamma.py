"""
Gamma probability distribution
------------------------------

The parameters used here are `k`, the shape parameter, and
`scale`, the scale parameter.

Another common parameterization is shape `k` and the "rate" `lambda`.
`lambda` is the reciprocal of `scale`.

"""

from mpmath import mp
from ._common import (_validate_p, _validate_moment_n, _validate_x_bounds,
                      _find_bracket)
from ..fun import digammainv
from .. import stats
from . import normal


__all__ = ['pdf', 'logpdf', 'cdf', 'invcdf', 'sf', 'invsf', 'interval_prob',
           'support',
           'mean', 'var', 'skewness', 'kurtosis', 'noncentral_moment',
           'entropy',
           'mom', 'mle',
           'nll', 'nll_grad', 'nll_hess', 'nll_invhess']


def _validate_k_scale(k, scale):
    if k <= 0:
        raise ValueError('k must be positive')
    if scale <= 0:
        raise ValueError('scale must be positive')
    return mp.mpf(k), mp.mpf(scale)


def pdf(x, k, scale):
    """
    Gamma distribution probability density function.

    `k` is the shape parameter
    `scale` is the scale parameter (reciprocal of the rate parameter)

    Unlike scipy, a location parameter is not included.
    """
    with mp.extradps(5):
        k, scale = _validate_k_scale(k, scale)
        x = mp.mpf(x)
        if x < 0:
            return mp.zero
        return mp.rgamma(k) / scale**k * x**(k-1) * mp.exp(-x/scale)


def logpdf(x, k, scale):
    """
    Log of the PDF of the gamma distribution.
    """
    with mp.extradps(5):
        k, scale = _validate_k_scale(k, scale)
        x = mp.mpf(x)
        if x < 0:
            return mp.ninf
        return (-mp.loggamma(k) - k*mp.log(scale) +
                (k - 1)*mp.log(x) - x/scale)


def cdf(x, k, scale):
    """
    Gamma distribution cumulative distribution function.

    `k` is the shape parameter
    `scale` is the scale parameter (reciprocal of the rate parameter)

    Unlike scipy, a location parameter is not included.
    """
    with mp.extradps(5):
        k, scale = _validate_k_scale(k, scale)
        x = mp.mpf(x)
        if x < 0:
            return mp.zero
        return mp.gammainc(k, 0, x/scale, regularized=True)


def invcdf(p, k, scale):
    """
    Inverse of the CDF of the gamma distribution.
    """
    with mp.extradps(max(10, mp.dps)):
        k, scale = _validate_k_scale(k, scale)
        p = _validate_p(p)
        if p == 0:
            return mp.zero
        if p == 1:
            return mp.inf

        x0, x1 = _find_bracket(lambda x: cdf(x, k, scale), p, 0, mp.inf,
                               nbisect=16)
        if x0 == x1:
            return x0
        x = mp.findroot(lambda x: cdf(x, k, scale) - p, x0=(x0, x1),
                        solver='secant')
        return x


def sf(x, k, scale):
    """
    Survival function of the gamma distribution.
    """
    with mp.extradps(5):
        k, scale = _validate_k_scale(k, scale)
        x = mp.mpf(x)
        if x < 0:
            return mp.one
        return mp.gammainc(k, x/scale, mp.inf, regularized=True)


def invsf(p, k, scale):
    """
    Inverse of the survival function of the gamma distribution.
    """
    with mp.extradps(max(10, mp.dps)):
        k, scale = _validate_k_scale(k, scale)
        p = _validate_p(p)
        if p == 0:
            return mp.inf
        if p == 1:
            return mp.zero

        x0, x1 = _find_bracket(lambda x: sf(x, k, scale), p, 0, mp.inf,
                               nbisect=16)
        if x0 == x1:
            return x0
        x = mp.findroot(lambda x: sf(x, k, scale) - p, x0=(x0, x1),
                        solver='secant')
        return x


def interval_prob(x1, x2, k, scale):
    """
    Compute the probability of x in [x1, x2] for the gamma distribution.

    Mathematically, this is the same as

        gamma.cdf(x2, k, scale) - gamma.cdf(x1, k, scale)

    but when the two CDF values are nearly equal, this function will give
    a more accurate result.

    x1 must be less than or equal to x2.
    """
    with mp.extradps(5):
        k, scale = _validate_k_scale(k, scale)
        x1 = mp.mpf(x1)
        x2 = mp.mpf(x2)
        if x1 > x2:
            raise ValueError('x1 must not be greater than x2')
        return mp.gammainc(k, x1/scale, x2/scale, regularized=True)


def support(k, scale):
    """
    Support of the gamma distribution.
    """
    with mp.extradps(5):
        k, scale = _validate_k_scale(k, scale)
        return (mp.zero, mp.inf)


def mean(k, scale):
    """
    Mean of the gamma distribution.
    """
    with mp.extradps(5):
        k, scale = _validate_k_scale(k, scale)
        return k * scale


def var(k, scale):
    """
    Variance of the gamma distribution.
    """
    with mp.extradps(5):
        k, scale = _validate_k_scale(k, scale)
        return k * scale**2


def skewness(k, scale):
    """
    Skewness of the gamma distribution.
    """
    with mp.extradps(5):
        k, scale = _validate_k_scale(k, scale)
        return 2/mp.sqrt(k)


def kurtosis(k, scale):
    """
    Excess kurtosis of the gamma distribution.
    """
    with mp.extradps(5):
        k, scale = _validate_k_scale(k, scale)
        return 6/k


def noncentral_moment(n, k, scale):
    """
    n-th noncentral moment of the gamma distribution.

    n must be a nonnegative integer.
    """
    n = _validate_moment_n(n)
    with mp.extradps(5):
        k, scale = _validate_k_scale(k, scale)
        if n == 0:
            return mp.one
        return scale**n * mp.gammaprod([k + n], [k])


def entropy(k, scale):
    """
    Differential entropy of the gamma distribution.
    """
    with mp.extradps(5):
        k, scale = _validate_k_scale(k, scale)
        return k + mp.log(scale) + mp.loggamma(k) + (1 - k)*mp.psi(0, k)


def mom(x):
    """
    Parameter estimation by the method of moments for the gamma distribution.

    x must be a sequence of values.

    Returns the estimates of the shape k and the scale.
    """
    with mp.extradps(5):
        m = stats.mean(x)
        v = stats.var(x)
        return m**2/v, v/m


def mle(x, *, k=None, scale=None):
    """
    Gamma distribution maximum likelihood parameter estimation.

    Maximum likelihood estimate for the k (shape) and scale parameters
    of the gamma distribution.

    x must be a sequence of values.
    """
    meanx = mp.fsum(x) / len(x)
    meanlnx = mp.fsum(mp.log(t) for t in x) / len(x)

    if k is None:
        if scale is None:
            # Solve for k and scale
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
            scale_hat = meanx / k_hat
        else:
            # scale is fixed, only solve for k
            scale = mp.mpf(scale)
            k_hat = digammainv(meanlnx - mp.log(scale))
            scale_hat = scale
    else:
        if scale is None:
            # Solve for scale, k is fixed.
            k_hat = mp.mpf(k)
            scale_hat = meanx / k_hat
        else:
            # Both k and scale are fixed.
            k_hat = mp.mpf(k)
            scale_hat = mp.mpf(scale)

    return k_hat, scale_hat


def nll(x, k, scale):
    """
    Gamma distribution negative log-likelihood.
    """
    with mp.extradps(5):
        k, scale = _validate_k_scale(k, scale)
        x = _validate_x_bounds(x, low=0, high=mp.inf,
                               strict_low=False, strict_high=True)
        N = len(x)
        sumx = mp.fsum(x)
        sumlnx = mp.fsum(mp.log(t) for t in x)

        ll = ((k - 1)*sumlnx - sumx/scale - N*k*mp.log(scale) -
              N*mp.loggamma(k))
        return -ll


def nll_grad(x, k, scale):
    """
    Gamma distribution gradient of the negative log-likelihood function.
    """
    with mp.extradps(5):
        k, scale = _validate_k_scale(k, scale)

        N = len(x)
        sumx = mp.fsum(x)
        sumlnx = mp.fsum(mp.log(t) for t in x)

        dk = sumlnx - N*mp.log(scale) - N*mp.digamma(k)
        dscale = sumx/scale**2 - N*k/scale
        return [-dk, -dscale]


def nll_hess(x, k, scale):
    """
    Gamma distribution hessian of the negative log-likelihood function.
    """
    with mp.extradps(5):
        k, scale = _validate_k_scale(k, scale)

        N = len(x)
        sumx = mp.fsum(x)
        # sumlnx = mp.fsum(mpmath.log(t) for t in x)

        dk2 = -N*mp.psi(1, k)
        dkdscale = -N/scale
        dscale2 = -2*sumx/scale**3 + N*k/scale**2

        return mp.matrix([[-dk2, -dkdscale], [-dkdscale, -dscale2]])


def nll_invhess(x, k, scale):
    """
    Gamma distribution inverse of the hessian of the negative log-likelihood.
    """
    with mp.extradps(5):
        k, scale = _validate_k_scale(k, scale)

        N = len(x)
        sumx = mp.fsum(x)
        # sumlnx = mp.fsum(mpmath.log(t) for t in x)

        dk2 = -N*mp.psi(1, k)
        dkdscale = -N/scale
        dscale2 = -2*sumx/scale**3 + N*k/scale**2

        det = dk2*dscale2 - dkdscale**2

        return mp.matrix([[-dscale2/det, dkdscale/det],
                          [dkdscale/det, -dk2/det]])


#
# The following are experimental and not thoroughly checked.
#

def _mle_se(x, k, scale):
    """
    Standard errors of the MLE estimates k and scale for the sequence x.

    This function assumes that both k and scale were estimated from x.
    """
    with mp.extradps(5):
        k, scale = _validate_k_scale(k, scale)
        invhess_diag = _nll_invhess_diag(x, k, scale)
        return (mp.sqrt(invhess_diag[0]), mp.sqrt(invhess_diag[1]))


def _mle_ci(x, k, scale, alpha):
    """
    Confidence intervals of the MLE estimates k and scale for the sequence x.

    This function assumes that both k and scale were estimated from x.

    The values are based on the asymptotic normality of the estimates.
    These are not the exact confidence intervals.
    """
    with mp.extradps(5):
        k, scale = _validate_k_scale(k, scale)
        kse, scalese = _mle_se(x, k, scale)
        w = normal.invcdf(alpha/2)
        return (k + w*kse, k - w*kse), (scale + w*scalese, scale - w*scalese)


def _nll_invhess_diag(x, k, scale):
    """
    Diagonal elements of the inverse of the hessian of the negative
    log-likelihood of the gamma distribution.
    """
    with mp.extradps(5):
        k, scale = _validate_k_scale(k, scale)

        N = len(x)
        sumx = mp.fsum(x)
        # sumlnx = mp.fsum(mp.log(t) for t in x)

        dk2 = -N*mp.psi(1, k)
        dkdscale = -N/scale
        dscale2 = -2*sumx/scale**3 + N*k/scale**2

        det = dk2*dscale2 - dkdscale**2

        return (-dscale2/det, -dk2/det)
