"""
Dirichlet distribution

https://en.wikipedia.org/wiki/Dirichlet_distribution
"""

from mpmath import mp
from ..fun import multivariate_logbeta, xlogy


__all__ = ['logpdf', 'pdf', 'mean', 'cov', 'entropy']


def _validate_alpha(alpha):
    for a in alpha:
        if a <= 0:
            raise ValueError('each element of alpha must be positive')
    return [mp.mpf(a) for a in alpha]


def _validate_x(x):
    for t in x:
        if t < 0:
            raise ValueError('each value in x must be nonnegative')
    x = [mp.mpf(t) for t in x]
    s = mp.fsum(x)
    if not mp.almosteq(s, 1):
        raise ValueError('sum(x) must be 1 (to within the tolerance '
                         f'of mp.almosteq), but got {s}')
    return x


def _validate_x_alpha(x, alpha):
    alpha = _validate_alpha(alpha)
    x = _validate_x(x)
    if len(alpha) != len(x):
        raise ValueError('x and alpha must be the same length')
    return x, alpha


def logpdf(x, alpha):
    """
    Logarithm of the PDF of the Dirichlet distribution.
    """
    x, alpha = _validate_x_alpha(x, alpha)
    with mp.extradps(5):
        b = multivariate_logbeta(alpha)
        terms = [xlogy(a - 1, t) for t, a in zip(x, alpha)]
        terms.append(-b)
        return mp.fsum(terms)


def pdf(x, alpha):
    """
    Probability density function of the Dirichlet distribution.
    """
    return mp.exp(logpdf(x, alpha))


def mean(alpha):
    """
    Mean of the Dirichlet distribution.
    """
    alpha = _validate_alpha(alpha)
    with mp.extradps(5):
        a0 = mp.fsum(alpha)
        return [a/a0 for a in alpha]


def cov(alpha):
    """
    Covariance matrix of the Dirichlet distribution.

    Returns an instance of `mpmath.matrix`.
    """
    alpha = _validate_alpha(alpha)
    n = len(alpha)
    with mp.extradps(5):
        covar = mp.matrix(n)
        a0 = mp.fsum(alpha)
        r = [a/a0 for a in alpha]
        for i in range(n):
            covar[i, i] = r[i]*(1 - r[i])/(a0 + 1)
            for j in range(i+1, n):
                covar[i, j] = -r[i]*r[j]/(a0 + 1)
                covar[j, i] = covar[i, j]
    return covar


def entropy(alpha):
    """
    Differential entropy of the Dirichlet distribution.
    """
    alpha = _validate_alpha(alpha)
    with mp.extradps(5):
        a0 = mp.fsum(alpha)
        lb = multivariate_logbeta(alpha)
        d = (a0 - len(alpha))*mp.digamma(a0)
        terms = [-(a - 1)*mp.digamma(a) for a in alpha]
        return mp.fsum([lb, d] + terms)
