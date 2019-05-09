"""
Generalized extreme value distribution
--------------------------------------

Note that the parameter xi used here has the opposite sign
of the corresponding shape parameter in `scipy.stats.genextreme`.
"""

import mpmath


__all__ = ['pdf', 'cdf', 'mean', 'var']


def pdf(x, xi, mu=0, sigma=1):
    """
    Generalized extreme value distribution probability density function.
    """
    xi = mpmath.mpf(xi)
    mu = mpmath.mpf(mu)
    sigma = mpmath.mpf(sigma)

    # Formula from wikipedia, which has a sign convention for xi that
    # is the opposite of scipy's shape parameter.
    if xi != 0:
        t = mpmath.power(1 + ((x - mu)/sigma)*xi, -1/xi)
    else:
        t = mpmath.exp(-(x - mu)/sigma)
    p = mpmath.power(t, xi+1) * mpmath.exp(-t) / sigma
    return p


def cdf(x, xi, mu=0, sigma=1):
    """
    Generalized extreme value distribution cumulative density function.
    """
    xi = mpmath.mpf(xi)
    mu = mpmath.mpf(mu)
    sigma = mpmath.mpf(sigma)

    # Formula from wikipedia, which has a sign convention for xi that
    # is the opposite of scipy's shape parameter.
    if xi != 0:
        t = mpmath.power(1 + ((x - mu)/sigma)*xi, -1/xi)
    else:
        t = mpmath.exp(-(x - mu)/sigma)
    return mpmath.exp(-t)


def mean(xi, mu=0, sigma=1):
    """
    Mean of the generalized extreme value distribution.
    """
    xi = mpmath.mpf(xi)
    mu = mpmath.mpf(mu)
    sigma = mpmath.mpf(sigma)

    if xi == 0:
        return mu + sigma * mpmath.euler
    elif xi < 1:
        g1 = mpmath.gamma(mpmath.mp.one - xi)
        return mu + sigma * (g1 - mpmath.mp.one)/xi
    else:
        return mpmath.inf


def var(xi, mu=0, sigma=1):
    """
    Variance of the generalized extreme value distribution.
    """
    xi = mpmath.mpf(xi)
    mu = mpmath.mpf(mu)
    sigma = mpmath.mpf(sigma)

    if xi == 0:
        return sigma**2 * mpmath.pi**2 / 6
    elif 2*xi < 1:
        g1 = mpmath.gamma(mpmath.mp.one - xi)
        g2 = mpmath.gamma(mpmath.mp.one - 2*xi)
        return sigma**2 * (g2 - g1**2) / xi**2
    else:
        return mpmath.inf
