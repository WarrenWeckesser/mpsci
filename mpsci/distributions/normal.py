"""
Normal distribution
-------------------
"""

from mpmath import mp
from ._common import _validate_loc_scale, _validate_p, _seq_to_mp


__all__ = ['pdf', 'logpdf', 'cdf', 'sf', 'invcdf', 'invsf',
           'support', 'entropy', 'mle']


@mp.extradps(5)
def pdf(x, mu=0, sigma=1):
    """
    Normal distribution probability density function.
    """
    # Defined here for consistency, but this is just mp.npdf
    mu, sigma = _validate_loc_scale(mu, sigma, scale_name='sigma')
    return mp.npdf(x, mu, sigma)


@mp.extradps(5)
def logpdf(x, mu=0, sigma=1):
    """
    Logarithm of the PDF of the normal distribution.
    """
    x = mp.mpf(x)
    mu, sigma = _validate_loc_scale(mu, sigma, scale_name='sigma')
    logp = (-mp.log(2*mp.pi)/2 - mp.log(sigma)
            - (x - mu)**2/(2*sigma**2))
    return logp


@mp.extradps(5)
def cdf(x, mu=0, sigma=1):
    """
    Normal distribution cumulative distribution function.
    """
    # Defined here for consistency, but this is just mp.ncdf
    mu, sigma = _validate_loc_scale(mu, sigma, scale_name='sigma')
    return mp.ncdf(x, mu, sigma)


@mp.extradps(5)
def sf(x, mu=0, sigma=1):
    """
    Normal distribution survival function.
    """
    x = mp.mpf(x)
    mu, sigma = _validate_loc_scale(mu, sigma, scale_name='sigma')
    return mp.ncdf(-x + 2*mu, mu, sigma)


def invcdf(p, mu=0, sigma=1):
    """
    Normal distribution inverse CDF.

    This function is also known as the quantile function or the percent
    point function.
    """
    with mp.extradps(mp.dps):
        p = _validate_p(p)
        mu, sigma = _validate_loc_scale(mu, sigma, scale_name='sigma')
        a = mp.erfinv(2*p - 1)
        x = mp.sqrt(2)*sigma*a + mu
        return x


def invsf(p, mu=0, sigma=1):
    """
    Inverse of the survival function of the normal distribution.
    """
    with mp.extradps(mp.dps):
        p = _validate_p(p)
        mu, sigma = _validate_loc_scale(mu, sigma, scale_name='sigma')
        a = mp.erfinv(1 - 2*p)
        x = mp.sqrt(2)*sigma*a + mu
        return x


@mp.extradps(5)
def support(mu=0, sigma=1):
    """
    Support of the normal distribution.
    """
    mu, sigma = _validate_loc_scale(mu, sigma, scale_name='sigma')
    return (mp.ninf, mp.inf)


@mp.extradps(5)
def entropy(mu=0, sigma=1):
    """
    Differential entropy of the normal distribution.
    """
    mu, sigma = _validate_loc_scale(mu, sigma, scale_name='sigma')
    return (mp.log(2*mp.pi) + 1)/2 + mp.log(sigma)


@mp.extradps(5)
def nll(x, mu=1, sigma=1):
    """
    Negative log-likelihood for the normal distribution.
    """
    x = _seq_to_mp(x)
    mu, sigma = _validate_loc_scale(mu, sigma, scale_name='sigma')
    return -mp.fsum([logpdf(t, mu, sigma) for t in x])


# XXX Add standard errors and confidence intervals for the fitted parameters.

@mp.extradps(5)
def mle(x):
    """
    Normal distribution maximum likelihood parameter estimation.

    Returns (mu, sigma).
    """
    x = _seq_to_mp(x)
    N = len(x)
    meanx = sum(x) / N
    var = sum((xi - meanx)**2 for xi in x) / N
    sigma = mp.sqrt(var)
    return meanx, sigma
