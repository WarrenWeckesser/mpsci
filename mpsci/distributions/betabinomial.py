"""
Beta-binomial distribution
--------------------------

See https://en.wikipedia.org/wiki/Beta-binomial_distribution

"""

from mpmath import mp
from ..fun import logbinomial, logbeta
from ._common import _validate_x_bounds, Initial


__all__ = ['pmf', 'logpmf', 'cdf', 'sf', 'mean', 'var', 'skewness', 'kurtosis',
           'nll', 'mle']


def _validate_params(n, a, b):
    if n != int(n):
        raise ValueError('n must be an integer')
    if n < 0:
        raise ValueError('n must be nonngative')
    if a <= 0:
        raise ValueError('a must be positive')
    if b <= 0:
        raise ValueError('b must be positive')
    return n, mp.mpf(a), mp.mpf(b)


def logpmf(k, n, a, b):
    """
    Logarithm of PMF of the beta-binomial distribution.
    """
    with mp.extradps(5):
        n, a, b = _validate_params(n, a, b)
        if k < 0:
            return mp.ninf
        if k > n:
            return mp.ninf
        t1 = logbinomial(n, k)
        t2 = logbeta(k + a, n - k + b)
        t3 = logbeta(a, b)
        return t1 + t2 - t3


def pmf(k, n, a, b):
    """
    Probability mass function of the beta-binomial distribution.
    """
    with mp.extradps(5):
        n, a, b = _validate_params(n, a, b)
        if k < 0:
            return mp.zero
        if k > n:
            return mp.zero
        return mp.exp(logpmf(k, n, a, b))


def cdf(k, n, a, b):
    """
    Cumulative distribution function of the beta-binomial distribution.

    This function uses ``mpmath.mp.hyp3f2(a1, a2, a3, b1, b2, z)``.
    According to the docstring of that function, "Evaluation for ``|z-1|``
    small can currently be inaccurate or slow for some parameter
    combinations".
    """
    with mp.extradps(5):
        n, a, b = _validate_params(n, a, b)
        if k < 0:
            return mp.zero
        if k >= n:
            return mp.one
        # This formula is from Wolfram Alpha:
        #   CDF[BetaBinomialDistribution[a, b, n], k]
        t1 = mp.beta(n + b - k - 1, a + k + 1)
        t2 = mp.hyp3f2(1, -n + k + 1, a + k + 1, k + 2, -n - b + k + 2, 1)
        c = 1 - (t1 * t2)/((n + 1)*mp.beta(a, b)*mp.beta(n - k, k + 2))
        return c


def sf(k, n, a, b):
    """
    Survival function of the beta-binomial distribution.

    This function uses ``mpmath.mp.hyp3f2(a1, a2, a3, b1, b2, z)``.
    According to the docstring of that function, "Evaluation for ``|z-1|``
    small can currently be inaccurate or slow for some parameter
    combinations".
    """
    with mp.extradps(5):
        n, a, b = _validate_params(n, a, b)
        if k < 0:
            return mp.one
        if k >= n:
            return mp.zero
        t1 = mp.beta(n + b - k - 1, a + k + 1)
        t2 = mp.hyp3f2(1, -n + k + 1, a + k + 1, k + 2, -n - b + k + 2, 1)
        c = (t1 * t2)/((n + 1)*mp.beta(a, b)*mp.beta(n - k, k + 2))
        return c


def mean(n, a, b):
    """
    Mean of the beta-binomial distribution.
    """
    with mp.extradps(5):
        n, a, b = _validate_params(n, a, b)
        return n*a / (a + b)


def var(n, a, b):
    """
    Variance of the beta-binomial distribution.
    """
    with mp.extradps(5):
        n, a, b = _validate_params(n, a, b)
        s = a + b
        return n*a*b*(s + n) / s**2 / (s + 1)


def skewness(n, a, b):
    """
    Skewness of the beta-bimonial distribution.
    """
    with mp.extradps(5):
        n, a, b = _validate_params(n, a, b)
        s = a + b
        f1 = (s + 2*n)*(b - a)/(s + 2)
        f2 = mp.sqrt((1 + s)/(n*a*b*(s + n)))
        return f1*f2


def kurtosis(n, a, b):
    """
    Excess kurtosis of the beta-binomial distribution.
    """
    with mp.extradps(5):
        n, a, b, = _validate_params(n, a, b)
        s = a + b
        denom = a*b*n*(s + 2)*(s + 3)*(s + n)
        kurt = ((s + 1)*(3*n**2*(a**2*(b + 2)
                                 + a*(b - 2)*b
                                 + 2*b**2)
                         + (a**2 - a*(4*b + 1) + (b - 1)*b)*s**2
                         + 3*n*(a**3*(b + 2) + 2*a**2*b**2
                                + a*b**3 + 2*b**3))) / denom
        return kurt - 3


def nll(x, n, a, b):
    """
    Negative log-likelihood of the beta-binomial distribution.

    `x` must be a sequence of nonnegative integers, with
    ``0 <= x[i] <= n``.
    """
    with mp.extradps(5):
        n, a, b = _validate_params(n, a, b)
        x = _validate_x_bounds(x, low=0, high=n,
                               strict_low=False, strict_high=False)
        if not all([mp.isint(t) for t in x]):
            raise ValueError('all values in x must be integers')
        return -mp.fsum([logpmf(t, n, a, b) for t in x])


def mle(x, *, n=None, a=None, b=None):
    """
    Maximum likelihood estimation for the beta-binomial distribution.

    This function does not estimate ``n``; a fixed value of ``n`` must
    be given.

    Returns (n, a, b), where each value is the maximum likelihood
    estimate of the parameter if it was not given in the function call.

    The function works best when a good initial guess is provided for the
    parameters to be fit. Use :class:`mpsci.distributions.Initial` to specify
    an initial guess for the parameter.
    """
    if n is None:
        raise ValueError('a value of n must be given.')
    if not mp.isint(n) or n < 1:
        raise ValueError('n must be a positive integer')
    with mp.extradps(5):
        x = _validate_x_bounds(x, low=0, high=n,
                               strict_low=False, strict_high=False)
        if not all([mp.isint(t) for t in x]):
            raise ValueError('all values in x must be integers')

        # XXX Figure out better default initial guesses for a and b.
        if a is None:
            a = Initial(1)
        if b is None:
            b = Initial(1)

        if not isinstance(a, Initial):
            if not isinstance(b, Initial):
                n, a, b = _validate_params(n, a, b)
            else:
                n, a, _ = _validate_params(n, a, 1)
        elif not isinstance(b, Initial):
            n, _, b = _validate_params(n, 1, b)
        else:
            n, _, _ = _validate_params(n, 1, 1)

        if isinstance(a, Initial) and isinstance(b, Initial):
            # Both parameters free

            def first_order_eqns(a, b):
                psia = mp.digamma(a)
                psib = mp.digamma(b)
                psiab = mp.digamma(a + b)
                psinab = mp.digamma(n + a + b)
                s1 = psiab - psinab
                sa = s1 - psia
                sb = s1 - psib
                eqa = mp.fsum([mp.digamma(t + a) + sa for t in x])
                eqb = mp.fsum([mp.digamma(n - t + b) + sb for t in x])
                return [eqa, eqb]

            est = mp.findroot(first_order_eqns,
                              x0=(mp.mpf(a.initial), mp.mpf(b.initial)))
            return n, est[0, 0], est[1, 0]

        elif isinstance(a, Initial):
            # a is free, b is fixed.

            def first_order_eqn(a):
                s = -mp.digamma(a) + mp.digamma(a + b) - mp.digamma(n + a + b)
                return mp.fsum([mp.digamma(t + a) + s for t in x])

            est = mp.findroot(first_order_eqn, x0=mp.mpf(a.initial))
            return n, est, b

        elif isinstance(b, Initial):
            # b is free, a is fixed

            def first_order_eqn(b):
                s = -mp.digamma(b) + mp.digamma(a + b) - mp.digamma(n + a + b)
                return mp.fsum([mp.digamma(n - t + b) + s for t in x])

            est = mp.findroot(first_order_eqn, x0=mp.mpf(b.initial))
            return n, a, est

        else:
            # All parameters fixed, nothing to do.
            return n, a, b
