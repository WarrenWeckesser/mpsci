"""
Beta-binomial distribution
--------------------------

See https://en.wikipedia.org/wiki/Beta-binomial_distribution

"""

from mpmath import mp
from ..fun import logbinomial, logbeta


__all__ = ['pmf', 'logpmf', 'cdf', 'sf', 'mean', 'var', 'skewness', 'kurtosis']


def _validate_params(n, a, b):
    if n != int(n):
        raise ValueError('n must be an integer')
    if n < 0:
        raise ValueError('n must bo nonngative')
    if a <= 0:
        raise ValueError('a must be positive')
    if b <= 0:
        raise ValueError('a must be positive')
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
