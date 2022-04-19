"""
Generalized extreme value distribution
--------------------------------------

Note that the parameter xi used here has the opposite sign
of the corresponding shape parameter in `scipy.stats.genextreme`.
"""

import mpmath


__all__ = ['pdf', 'logpdf', 'cdf', 'sf', 'mean', 'var', 'skewness',
           'kurtosis']


def _validate_sigma(sigma):
    if sigma <= 0:
        raise ValueError('sigma must be positive')


def pdf(x, xi, mu=0, sigma=1):
    """
    Generalized extreme value distribution probability density function.
    """
    _validate_sigma(sigma)
    xi = mpmath.mpf(xi)
    mu = mpmath.mpf(mu)
    sigma = mpmath.mpf(sigma)

    # Formula from wikipedia, which has a sign convention for xi that
    # is the opposite of scipy's shape parameter.
    z = (x - mu)/sigma
    if xi != 0:
        t = mpmath.power(1 + z*xi, -1/xi)
    else:
        t = mpmath.exp(-z)
    p = mpmath.power(t, xi+1) * mpmath.exp(-t) / sigma
    return p


def logpdf(x, xi, mu=0, sigma=1):
    """
    Natural logarithm of the PDF of the generalized extreme value distribution.
    """
    _validate_sigma(sigma)
    xi = mpmath.mpf(xi)
    mu = mpmath.mpf(mu)
    sigma = mpmath.mpf(sigma)

    # Formula from wikipedia, which has a sign convention for xi that
    # is the opposite of scipy's shape parameter.
    z = (x - mu)/sigma
    if xi != 0:
        t = mpmath.power(1 + z*xi, -1/xi)
        logt = -mpmath.log1p(z*xi)/xi
    else:
        t = mpmath.exp(-z)
        logt = -z
    p = (xi + 1)*logt - t - mpmath.log(sigma)
    return p


def cdf(x, xi, mu=0, sigma=1):
    """
    Generalized extreme value distribution cumulative density function.
    """
    _validate_sigma(sigma)
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


def sf(x, xi, mu=0, sigma=1):
    """
    Generalized extreme value distribution survival function.
    """
    _validate_sigma(sigma)
    xi = mpmath.mpf(xi)
    mu = mpmath.mpf(mu)
    sigma = mpmath.mpf(sigma)

    # Formula from wikipedia, which has a sign convention for xi that
    # is the opposite of scipy's shape parameter.
    if xi != 0:
        t = mpmath.power(1 + ((x - mu)/sigma)*xi, -1/xi)
    else:
        t = mpmath.exp(-(x - mu)/sigma)
    return -mpmath.expm1(-t)


def mean(xi, mu=0, sigma=1):
    """
    Mean of the generalized extreme value distribution.
    """
    _validate_sigma(sigma)
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
    _validate_sigma(sigma)
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


def skewness(xi, mu=0, sigma=1):
    """
    Skewness of the generalized extreme value distribution.
    """
    _validate_sigma(sigma)
    xi = mpmath.mpf(xi)
    mu = mpmath.mpf(mu)
    sigma = mpmath.mpf(sigma)

    if xi == 0:
        return 12*mpmath.sqrt(6)*mpmath.zeta(3) / mpmath.pi**3
    elif 3*xi < 1:
        g1 = mpmath.gamma(mpmath.mp.one - xi)
        g2 = mpmath.gamma(mpmath.mp.one - 2*xi)
        g3 = mpmath.gamma(mpmath.mp.one - 3*xi)
        num = g3 - 3*g2*g1 + 2*g1**3
        den = mpmath.power(g2 - g1**2, 1.5)
        return mpmath.sign(xi)*num/den
    else:
        return mpmath.inf


def kurtosis(xi, mu=0, sigma=1):
    """
    Excess kurtosis of the generalized extreme value distribution.
    """
    _validate_sigma(sigma)
    xi = mpmath.mpf(xi)
    mu = mpmath.mpf(mu)
    sigma = mpmath.mpf(sigma)

    if xi == 0:
        return mpmath.mpf(12)/5
    elif 4*xi < 1:
        g1 = mpmath.gamma(mpmath.mp.one - xi)
        g2 = mpmath.gamma(mpmath.mp.one - 2*xi)
        g3 = mpmath.gamma(mpmath.mp.one - 3*xi)
        g4 = mpmath.gamma(mpmath.mp.one - 4*xi)
        num = g4 - 4*g3*g1 - 3*g2**2 + 12*g2*g1**2 - 6*g1**4
        den = (g2 - g1**2)**2
        return num/den
    else:
        return mpmath.inf
