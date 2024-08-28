"""
Generalized extreme value distribution
--------------------------------------

The parametrization used here is the same as that of the
wikipedia article "Generalized extreme value distribution"
(https://en.wikipedia.org/wiki/Generalized_extreme_value_distribution).

`xi` is a shape parameter, `mu` is a location parameter, and `sigma`
is a scale parameter.

Note that the parameter `xi` used here has the opposite sign
of the corresponding shape parameter `c` in `scipy.stats.genextreme`.

"""

from mpmath import mp
from ._common import _validate_p, _validate_moment_n


__all__ = ['pdf', 'logpdf', 'cdf', 'invcdf', 'sf', 'invsf',
           'mean', 'var', 'skewness', 'kurtosis', 'entropy',
           'noncentral_moment']


def _validate_params(xi, mu, sigma):
    if sigma <= 0:
        raise ValueError('sigma must be positive')
    return mp.mpf(xi), mp.mpf(mu), mp.mpf(sigma)


def pdf(x, xi, mu=0, sigma=1):
    """
    Probability density function of the generalized extreme value distribution.
    """
    with mp.extradps(5):
        xi, mu, sigma = _validate_params(xi, mu, sigma)

        if xi != 0:
            bound = mu - sigma/xi
            if xi > 0 and x <= bound:
                return mp.zero
            if xi < 0 and x >= bound:
                return mp.zero

        # Formula from wikipedia, which has a sign convention for xi that
        # is the opposite of scipy's shape parameter.
        z = (x - mu)/sigma
        if xi != 0:
            t = mp.power(1 + z*xi, -1/xi)
        else:
            t = mp.exp(-z)
        p = mp.power(t, xi + 1) * mp.exp(-t) / sigma
        return p


def logpdf(x, xi, mu=0, sigma=1):
    """
    Natural logarithm of the PDF of the generalized extreme value distribution.
    """
    with mp.extradps(5):
        xi, mu, sigma = _validate_params(xi, mu, sigma)

        if xi != 0:
            bound = mu - sigma/xi
            if xi > 0 and x <= bound:
                return mp.ninf
            if xi < 0 and x >= bound:
                return mp.ninf

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
    Cumulative distribution function of the generalized extreme value distribution.
    """
    with mp.extradps(5):
        xi, mu, sigma = _validate_params(xi, mu, sigma)

        if xi != 0:
            bound = mu - sigma/xi
            if xi > 0 and x <= bound:
                return mp.zero
            if xi < 0 and x >= bound:
                return mp.one

        # Formula from wikipedia, which has a sign convention for xi that
        # is the opposite of scipy's shape parameter.
        if xi != 0:
            t = mp.power(1 + ((x - mu)/sigma)*xi, -1/xi)
        else:
            t = mp.exp(-(x - mu)/sigma)
        return mp.exp(-t)


def invcdf(p, xi, mu=0, sigma=1):
    """
    Inverse of the CDF of the generalized extreme value distribution.
    """
    with mp.extradps(5):
        p = _validate_p(p)
        xi, mu, sigma = _validate_params(xi, mu, sigma)
        if xi != 0:
            x = mu + (sigma/xi)*mp.powm1(-mp.log(p), -xi)
        else:
            x = mu - sigma*mp.log(-mp.log(p))
        return x


def sf(x, xi, mu=0, sigma=1):
    """
    Survival function of the generalized extreme value distribution.
    """
    with mp.extradps(5):
        xi, mu, sigma = _validate_params(xi, mu, sigma)

        if xi != 0:
            bound = mu - sigma/xi
            if xi > 0 and x <= bound:
                return mp.one
            if xi < 0 and x >= bound:
                return mp.zero

        # Formula from wikipedia, which has a sign convention for xi that
        # is the opposite of scipy's shape parameter.
        if xi != 0:
            t = mp.power(1 + ((x - mu)/sigma)*xi, -1/xi)
        else:
            t = mp.exp(-(x - mu)/sigma)
        return -mp.expm1(-t)


def invsf(p, xi, mu=0, sigma=1):
    """
    Inverse of the survival function of the generalized extreme value dist.
    """
    with mp.extradps(5):
        p = _validate_p(p)
        xi, mu, sigma = _validate_params(xi, mu, sigma)
        if xi != 0:
            x = mu + (sigma/xi)*mp.powm1(-mp.log1p(-p), -xi)
        else:
            x = mu - sigma*mp.log(-mp.log1p(-p))
        return x


def support(xi, mu=0, sigma=1):
    """
    Support of the generalized extreme value distribution.

    The support depends on the parameters as follows:

    * xi < 0:  (-inf, mu - sigma/xi]
    * xi == 0: (-inf, inf)
    * xi > 0:  [mu - sigma/xi, inf)

    """
    with mp.extradps(5):
        xi, mu, sigma = _validate_params(xi, mu, sigma)
        if xi != 0:
            bound = mu - sigma/xi
            if xi > 0:
                return (bound, mp.inf)
            else:
                return (mp.ninf, bound)
        # xi == 0
        return (mp.ninf, mp.inf)


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


def entropy(xi, mu=0, sigma=1):
    """
    Differential entropy of the generalized extreme value distribution.
    """
    with mp.extradps(5):
        xi, mu, sigma = _validate_params(xi, mu, sigma)
        return mp.log(sigma) + mp.euler*(xi + 1) + 1


def _standard_noncentral_moment(n, xi):
    with mp.extradps(5):
        if n == 0:
            return mp.one
        if xi < mp.one/n:
            v = [mp.binomial(n, k) * (-1)**k * mp.gamma(1 - xi*k)
                 for k in range(n+1)]
            return mp.fsum(v)/(-xi)**n
        else:
            return mp.nan


def noncentral_moment(n, xi, mu=0, sigma=1):
    """
    Noncentral moment of the generalized extreme value distribution.

    The value is also known as the raw moment.
    """
    # Except for the check xi >= 1/n, this is a generic calculation that
    # could be applied to any loc/scale family if there is a function for
    # the standard (i.e. loc=0, scale=1) noncentral moment.
    with mp.extradps(5):
        _validate_moment_n(n)
        xi, mu, sigma = _validate_params(xi, mu, sigma)
        if n == 0:
            return mp.one
        if xi >= mp.one/n:
            # Maybe return inf if xi == 1/n?
            return mp.nan
        terms = [(mp.binomial(n, k) * mp.power(mu, n - k) * mp.power(sigma, k)
                  * _standard_noncentral_moment(k, xi))
                 for k in range(n + 1)]
        return mp.fsum(terms)
