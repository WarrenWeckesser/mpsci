"""
Generalized Pareto distribution
-------------------------------

"""

from mpmath import mp
from ._common import _validate_p, _validate_x_bounds
from ..fun import pow1pm1


__all__ = ['pdf', 'logpdf', 'cdf', 'invcdf', 'sf', 'invsf',
           'mean', 'var', 'entropy', 'nll']


def _validate_params(xi, mu, sigma):
    if sigma <= 0:
        raise ValueError('sigma must be greater than 0')
    return mp.mpf(xi), mp.mpf(mu), mp.mpf(sigma)


def pdf(x, xi, mu=0, sigma=1):
    """
    Generalized Pareto distribution probability density function.
    """
    with mp.extradps(5):
        xi, mu, sigma = _validate_params(xi, mu, sigma)
        x = mp.mpf(x)
        z = (x - mu)/sigma
        if (xi >= 0 and z < 0) or (xi < 0 and (z < 0 or z > -1/xi)):
            p = 0
        else:
            if xi != 0:
                t = mp.power(1 + z*xi, -(1/xi + 1))
            else:
                t = mp.exp(-z)
            p = t / sigma
        return p


def logpdf(x, xi, mu=0, sigma=1):
    """
    Natural logarithm of the PDF of the generalized Pareto distribution.
    """
    with mp.extradps(5):
        xi, mu, sigma = _validate_params(xi, mu, sigma)
        x = mp.mpf(x)
        z = (x - mu)/sigma
        if (xi >= 0 and z < 0) or (xi < 0 and (z < 0 or z > -1/xi)):
            p = -mp.inf
        else:
            if xi != 0:
                logt = -(1/xi + 1)*mp.log(1 + xi*z)
            else:
                logt = -z
            p = logt - mp.log(sigma)
        return p


def cdf(x, xi, mu=0, sigma=1):
    """
    Generalized Pareto distribution cumulative density function.
    """
    with mp.extradps(5):
        xi, mu, sigma = _validate_params(xi, mu, sigma)
        x = mp.mpf(x)
        z = (x - mu)/sigma
        if (xi >= 0 and z < 0) or (xi < 0 and z < 0):
            t = mp.zero
        elif xi < 0 and z > -1/xi:
            t = mp.one
        else:
            if xi != 0:
                t = -mp.powm1(1 + xi*z, -1/xi)
            else:
                t = -mp.expm1(-z)
        return t


def invcdf(p, xi, mu=0, sigma=1):
    """
    Inverse of the CDF of the generalized Pareto distribution.
    """
    with mp.extradps(5):
        xi, mu, sigma = _validate_params(xi, mu, sigma)
        p = _validate_p(p)

        # Handle end points.
        if p == 0:
            return mu
        if p == 1:
            if xi >= 0:
                return mp.inf
            else:
                return mu - sigma/xi

        if xi == 0:
            return mu - sigma*mp.log1p(-p)
        else:
            # return mu + (sigma/xi)*((1 - p)**(-xi) - 1)
            return mu + (sigma/xi)*pow1pm1(-p, -xi)


def sf(x, xi, mu=0, sigma=1):
    """
    Generalized Pareto distribution survival function.
    """
    with mp.extradps(5):
        xi, mu, sigma = _validate_params(xi, mu, sigma)
        x = mp.mpf(x)
        z = (x - mu)/sigma
        if (xi >= 0 and z < 0) or (xi < 0 and z < 0):
            t = mp.one
        elif xi < 0 and z > -1/xi:
            t = mp.zero
        else:
            if xi != 0:
                t = mp.power(1 + xi*z, -1/xi)
            else:
                t = mp.exp(-z)
        return t


def invsf(p, xi, mu=0, sigma=1):
    """
    Inverse of the survival function of the generalized Pareto distribution.
    """
    with mp.extradps(5):
        xi, mu, sigma = _validate_params(xi, mu, sigma)
        p = _validate_p(p)

        # Handle end points.
        if p == 1:
            return mu
        if p == 0:
            if xi >= 0:
                return mp.inf
            else:
                return mu - sigma/xi

        if xi == 0:
            return mu - sigma*mp.log(p)
        else:
            return mu + (sigma/xi)*mp.powm1(p, -xi)


def mean(xi, mu=0, sigma=1):
    """
    Mean of the generalized Pareto distribution.
    """
    with mp.extradps(5):
        xi, mu, sigma = _validate_params(xi, mu, sigma)
        if xi < 1:
            return mu + sigma / (1 - xi)
        return mp.nan


def var(xi, mu=0, sigma=1):
    """
    Variance of the generalized Pareto distribution.
    """
    with mp.extradps(5):
        xi, mu, sigma = _validate_params(xi, mu, sigma)
        if xi < 0.5:
            return sigma**2 / (1 - xi)**2 / (1 - 2*xi)
        return mp.nan


def entropy(xi, mu=0, sigma=1):
    """
    Entropy of the generalized Pareto distribution.
    """
    with mp.extradps(5):
        xi, mu, sigma = _validate_params(xi, mu, sigma)
        return mp.log(sigma) + xi + 1


def nll(x, xi, mu=0, sigma=1):
    """
    Negative log-likelihood function for the generalized Pareto distribution.
    """
    with mp.extradps(5):
        xi, mu, sigma = _validate_params(xi, mu, sigma)
        if xi >= 0:
            high = mp.inf
            strict_high = True
            highname = None
        else:
            high = mu - sigma/xi
            strict_high = False
            highname = 'mu - sigma/xi'
        x = _validate_x_bounds(x, low=mu, strict_low=False, lowname='mu',
                               high=high, strict_high=strict_high,
                               highname=highname)
        return -mp.fsum([logpdf(t, xi, mu=mu, sigma=sigma) for t in x])
