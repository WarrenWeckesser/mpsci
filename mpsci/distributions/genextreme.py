"""
Generalized extreme value distribution
--------------------------------------

Note that the parameter xi used here has the opposite sign
of the corresponding shape parameter in `scipy.stats.genextreme`.
"""

from mpmath import mp


__all__ = ['pdf', 'logpdf', 'cdf', 'sf', 'mean', 'var', 'skewness',
           'kurtosis']


def _validate_params(xi, mu, sigma):
    if sigma <= 0:
        raise ValueError('sigma must be positive')
    return mp.mpf(xi), mp.mpf(mu), mp.mpf(sigma)


def pdf(x, xi, mu=0, sigma=1):
    """
    Generalized extreme value distribution probability density function.
    """
    with mp.extradps(5):
        xi, mu, sigma = _validate_params(xi, mu, sigma)

        # Formula from wikipedia, which has a sign convention for xi that
        # is the opposite of scipy's shape parameter.
        z = (x - mu)/sigma
        if xi != 0:
            t = mp.power(1 + z*xi, -1/xi)
        else:
            t = mp.exp(-z)
        p = mp.power(t, xi+1) * mp.exp(-t) / sigma
        return p


def logpdf(x, xi, mu=0, sigma=1):
    """
    Natural logarithm of the PDF of the generalized extreme value distribution.
    """
    with mp.extradps(5):
        xi, mu, sigma = _validate_params(xi, mu, sigma)
        # Formula from wikipedia, which has a sign convention for xi that
        # is the opposite of scipy's shape parameter.
        z = (x - mu)/sigma
        if xi != 0:
            t = mp.power(1 + z*xi, -1/xi)
            logt = -mp.log1p(z*xi)/xi
        else:
            t = mp.exp(-z)
            logt = -z
        p = (xi + 1)*logt - t - mp.log(sigma)
        return p


def cdf(x, xi, mu=0, sigma=1):
    """
    Generalized extreme value distribution cumulative density function.
    """
    with mp.extradps(5):
        xi, mu, sigma = _validate_params(xi, mu, sigma)
        # Formula from wikipedia, which has a sign convention for xi that
        # is the opposite of scipy's shape parameter.
        if xi != 0:
            t = mp.power(1 + ((x - mu)/sigma)*xi, -1/xi)
        else:
            t = mp.exp(-(x - mu)/sigma)
        return mp.exp(-t)


def sf(x, xi, mu=0, sigma=1):
    """
    Generalized extreme value distribution survival function.
    """
    with mp.extradps(5):
        xi, mu, sigma = _validate_params(xi, mu, sigma)
        # Formula from wikipedia, which has a sign convention for xi that
        # is the opposite of scipy's shape parameter.
        if xi != 0:
            t = mp.power(1 + ((x - mu)/sigma)*xi, -1/xi)
        else:
            t = mp.exp(-(x - mu)/sigma)
        return -mp.expm1(-t)


def mean(xi, mu=0, sigma=1):
    """
    Mean of the generalized extreme value distribution.
    """
    with mp.extradps(5):
        xi, mu, sigma = _validate_params(xi, mu, sigma)
        if xi == 0:
            return mu + sigma * mp.euler
        elif xi < 1:
            g1 = mp.gamma(mp.one - xi)
            return mu + sigma * (g1 - mp.one)/xi
        else:
            return mp.inf


def var(xi, mu=0, sigma=1):
    """
    Variance of the generalized extreme value distribution.
    """
    with mp.extradps(5):
        xi, mu, sigma = _validate_params(xi, mu, sigma)
        if xi == 0:
            return sigma**2 * mp.pi**2 / 6
        elif 2*xi < 1:
            g1 = mp.gamma(mp.one - xi)
            g2 = mp.gamma(mp.one - 2*xi)
            return sigma**2 * (g2 - g1**2) / xi**2
        else:
            return mp.inf


def skewness(xi, mu=0, sigma=1):
    """
    Skewness of the generalized extreme value distribution.
    """
    with mp.extradps(5):
        xi, mu, sigma = _validate_params(xi, mu, sigma)
        if xi == 0:
            return 12*mp.sqrt(6)*mp.zeta(3) / mp.pi**3
        elif 3*xi < 1:
            g1 = mp.gamma(mp.one - xi)
            g2 = mp.gamma(mp.one - 2*xi)
            g3 = mp.gamma(mp.one - 3*xi)
            num = g3 - 3*g2*g1 + 2*g1**3
            den = mp.power(g2 - g1**2, 1.5)
            return mp.sign(xi)*num/den
        else:
            return mp.inf


def kurtosis(xi, mu=0, sigma=1):
    """
    Excess kurtosis of the generalized extreme value distribution.
    """
    with mp.extradps(5):
        xi, mu, sigma = _validate_params(xi, mu, sigma)
        if xi == 0:
            return mp.mpf(12)/5
        elif 4*xi < 1:
            g1 = mp.gamma(mp.one - xi)
            g2 = mp.gamma(mp.one - 2*xi)
            g3 = mp.gamma(mp.one - 3*xi)
            g4 = mp.gamma(mp.one - 4*xi)
            num = g4 - 4*g3*g1 - 3*g2**2 + 12*g2*g1**2 - 6*g1**4
            den = (g2 - g1**2)**2
            return num/den
        else:
            return mp.inf
