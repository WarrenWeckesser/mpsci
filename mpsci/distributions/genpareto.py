"""
Generalized Pareto distribution
-------------------------------

Note that the parameter xi used here has the opposite sign
of the corresponding shape parameter in `scipy.stats.genextreme`.
"""

import mpmath


__all__ = ['pdf', 'logpdf', 'cdf', 'sf']


def pdf(x, xi, mu=0, sigma=1):
    """
    Generalized Pareto distribution probability density function.
    """
    with mpmath.extradps(5):
        xi = mpmath.mpf(xi)
        mu = mpmath.mpf(mu)
        sigma = mpmath.mpf(sigma)

        z = (x - mu)/sigma
        if xi != 0:
            t = mpmath.power(1 + z*xi, -(1/xi + 1))
        else:
            t = mpmath.exp(-z)
        p = t / sigma
    return p


def logpdf(x, xi, mu=0, sigma=1):
    """
    Natural logarithm of the PDF of the generalized Pareto distribution.
    """
    with mpmath.extradps(5):
        xi = mpmath.mpf(xi)
        mu = mpmath.mpf(mu)
        sigma = mpmath.mpf(sigma)

        z = (x - mu)/sigma
        if xi != 0:
            logt = -(1/xi + 1)*mpmath.log(1 + xi*z)
        else:
            logt = -z
        p = logt - mpmath.log(sigma)
    return p


def cdf(x, xi, mu=0, sigma=1):
    """
    Generalized Pareto distribution cumulative density function.
    """
    with mpmath.extradps(5):
        xi = mpmath.mpf(xi)
        mu = mpmath.mpf(mu)
        sigma = mpmath.mpf(sigma)

        z = (x - mu)/sigma
        if xi != 0:
            t = 1 - mpmath.power(1 + xi*z, -1/xi)
        else:
            t = 1 - mpmath.exp(-z)
    return t


def sf(x, xi, mu=0, sigma=1):
    """
    Generalized Pareto distribution survival function.
    """
    with mpmath.extradps(5):
        xi = mpmath.mpf(xi)
        mu = mpmath.mpf(mu)
        sigma = mpmath.mpf(sigma)

        z = (x - mu)/sigma
        if xi != 0:
            t = mpmath.power(1 + xi*z, -1/xi)
        else:
            t = mpmath.exp(-z)
    return t
