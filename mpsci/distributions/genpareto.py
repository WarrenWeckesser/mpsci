"""
Generalized Pareto distribution
-------------------------------

"""

import mpmath


__all__ = ['pdf', 'logpdf', 'cdf', 'sf', 'mean', 'var', 'entropy']


def pdf(x, xi, mu=0, sigma=1):
    """
    Generalized Pareto distribution probability density function.
    """
    with mpmath.extradps(5):
        xi = mpmath.mpf(xi)
        mu = mpmath.mpf(mu)
        sigma = mpmath.mpf(sigma)

        z = (x - mu)/sigma
        if (xi >= 0 and z < 0) or (xi < 0 and (z < 0 or z > -1/xi)):
            p = 0
        else:
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
        if (xi >= 0 and z < 0) or (xi < 0 and (z < 0 or z > -1/xi)):
            p = -mpmath.mp.inf
        else:
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
        if (xi >= 0 and z < 0) or (xi < 0 and z < 0):
            t = mpmath.mp.zero
        elif xi < 0 and z > -1/xi:
            t = mpmath.mp.one
        else:
            if xi != 0:
                t = -mpmath.powm1(1 + xi*z, -1/xi)
            else:
                t = -mpmath.expm1(-z)
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
        if (xi >= 0 and z < 0) or (xi < 0 and z < 0):
            t = mpmath.mp.one
        elif xi < 0 and z > -1/xi:
            t = mpmath.mp.zero
        else:
            if xi != 0:
                t = mpmath.power(1 + xi*z, -1/xi)
            else:
                t = mpmath.exp(-z)
    return t


def mean(xi, mu=0, sigma=1):
    """
    Mean of the generalized Pareto distribution.
    """
    with mpmath.extradps(5):
        xi = mpmath.mpf(xi)
        mu = mpmath.mpf(mu)
        sigma = mpmath.mpf(sigma)

        if xi < 1:
            return mu + sigma / (1 - xi)
        return mpmath.mp.nan


def var(xi, mu=0, sigma=1):
    """
    Variance of the generalized Pareto distribution.
    """
    with mpmath.extradps(5):
        xi = mpmath.mpf(xi)
        mu = mpmath.mpf(mu)
        sigma = mpmath.mpf(sigma)

        if xi < 0.5:
            return sigma**2 / (1 - xi)**2 / (1 - 2*xi)
        return mpmath.mp.nan


def entropy(xi, mu=0, sigma=1):
    """
    Entropy of the generalized Pareto distribution.
    """
    with mpmath.extradps(5):
        xi = mpmath.mpf(xi)
        mu = mpmath.mpf(mu)
        sigma = mpmath.mpf(sigma)

        return mpmath.log(sigma) + xi + 1
