"""
Logistic distribution
---------------------

The logistic distribution is also known as the sech-squared distribution.

"""

from mpmath import mp
from mpsci.stats import mean as _mean
from ._common import _seq_to_mp


__all__ = ['pdf', 'logpdf', 'cdf', 'sf', 'invcdf', 'invsf',
           'support', 'mean', 'var',
           'mom', 'nll', 'mle']


def pdf(x, loc=0, scale=1):
    """
    PDF of the logistic distribution.
    """
    with mp.extradps(5):
        x = mp.mpf(x)
        loc = mp.mpf(loc)
        scale = mp.mpf(scale)
        z = (x - loc) / scale
        p = mp.sech(z/2)**2 / (4*scale)
    return p


def logpdf(x, loc=0, scale=1):
    """
    Logarithm of the PDF of the logistic distribution.
    """
    with mp.extradps(5):
        x = mp.mpf(x)
        loc = mp.mpf(loc)
        scale = mp.mpf(scale)
        z = (x - loc) / scale
        logp = 2*mp.log(mp.sech(z/2)) - mp.log(4*scale)
    return logp


def cdf(x, loc=0, scale=1):
    """
    CDF of the logistic distribution.
    """
    with mp.extradps(5):
        x = mp.mpf(x)
        loc = mp.mpf(loc)
        scale = mp.mpf(scale)
        z = (x - loc) / scale
        p = (1 + mp.tanh(z/2)) / 2
    return p


def sf(x, loc=0, scale=1):
    """
    Survival function of the logistic distribution.
    """
    with mp.extradps(5):
        x = mp.mpf(x)
        loc = mp.mpf(loc)
        scale = mp.mpf(scale)
        z = (x - loc) / scale
        p = (1 - mp.tanh(z/2)) / 2
    return p


def invcdf(p, loc=0, scale=1):
    """
    Inverse CDF of the logistic distribution.

    This function is also known as the quantile function or the percent
    point function.
    """
    with mp.extradps(5):
        p = mp.mpf(p)
        loc = mp.mpf(loc)
        scale = mp.mpf(scale)
        x = loc + scale*(mp.log(p) - mp.log1p(-p))
    return x


def invsf(p, loc=0, scale=1):
    """
    Inverse survival function of the logistic distribution.
    """
    with mp.extradps(5):
        p = mp.mpf(p)
        loc = mp.mpf(loc)
        scale = mp.mpf(scale)
        x = loc + scale*(mp.log1p(-p) - mp.log(p))
    return x


def support(loc=0, scale=1):
    """
    Support of the logistic distribution.
    """
    with mp.extradps(5):
        return (mp.ninf, mp.inf)


def mean(loc=0, scale=1):
    """
    Mean of the logistic distribution.
    """
    with mp.extradps(5):
        return mp.mpf(loc)


def var(loc=0, scale=1):
    """
    Variance of the logistic distribution.
    """
    with mp.extradps(5):
        scale = mp.mpf(scale)
        return scale**2 * mp.pi**2 / 3


def entropy(loc=0, scale=1):
    """
    Differential entropy of the logistic distribution.
    """
    with mp.extradps(5):
        scale = mp.mpf(scale)
        return mp.log(scale) + 2


def mom(x):
    """
    Method of moments parameter estimation for the logistic distribution.

    `x` must be a sequence of numbers.

    Returns (loc, scale).
    """
    with mp.extradps(5):
        x = _seq_to_mp(x)
        M1 = _mean(x)
        M2 = _mean([t**2 for t in x])
        return M1, mp.sqrt(3*(M2 - M1**2))/mp.pi


def nll(x, loc, scale):
    """
    Negative log-likelihood function for the logistic equation.

    `x` must be a sequence of numbers.
    """
    with mp.extradps(5):
        x = _seq_to_mp(x)
        loc = mp.mpf(loc)
        scale = mp.mpf(scale)
        v = [mp.log(mp.sech((t - loc)/(2*scale))) for t in x]
        n = len(x)
        return n*mp.log(4*scale) - 2*mp.fsum(v)


def _mle_loc_eq(loc, scale, x):
    v = [mp.tanh((t - loc)/(2*scale)) for t in x]
    return mp.fsum(v)


def _mle_scale_eq(loc, scale, x):
    n = len(x)
    v = [t*mp.tanh((t - loc)/(2*scale)) for t in x]
    return -n + mp.fsum(v)/scale


def mle(x):
    """
    Maximum likelihood estimate for the logistic distribution.

    `x` must be a sequence of numbers.

    Returns (loc, scale).

    This function uses `mp.findroot` to numerically solve for the
    maximum likelihood estimate.
    """
    with mp.extradps(mp.dps//4):
        loc0, scale0 = mom(x)
        x = _seq_to_mp(x)
        loc1, scale1 = mp.findroot(
            lambda loc, scale: [_mle_loc_eq(loc, scale, x),
                                _mle_scale_eq(loc, scale, x)],
            x0=[loc0, scale0]
        )
        # Because of the symmetry with respect to scale of the first order
        # equations, `findroot` might return a negative scale.
        # If (loc1, scale1) solves the equations, then so does (loc1, -scale1),
        # so we can return the absolute value of scale1.
        return loc1, abs(scale1)
