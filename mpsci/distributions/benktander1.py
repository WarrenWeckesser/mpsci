"""
Benktander I Distribution
-------------------------
"""

from mpmath import mp
from ..stats import mean as _mean
from ._common import _validate_p, _validate_x_bounds, Initial


__all__ = ['pdf', 'logpdf', 'cdf', 'sf', 'invcdf', 'invsf',
           'support', 'mean', 'var', 'nll']


def _validate_ab(a, b):
    if a <= 0:
        raise ValueError("'a' must be positive.")
    if b <= 0:
        raise ValueError("'b' must be positive.")
    if b > a*(a + 1)/2:
        raise ValueError("'b' must not be greater than a*(a+1)/2.")
    return mp.mpf(a), mp.mpf(b)


def pdf(x, a, b):
    """
    PDF of the Benktander I distribution.

    Variable names follow the convention used on wikipedia.
    """
    with mp.extradps(5):
        a, b = _validate_ab(a, b)
        x = mp.mpf(x)
        if x < 1:
            return mp.zero
        blogx = b*mp.log(x)
        c = (1 + 2*blogx/a)*(1 + a + 2*blogx) - 2*b/a
        return c * mp.power(x, -(2 + a + blogx))


def logpdf(x, a, b):
    """
    Logarithm of the PDF of the Benktander I distribution.

    Variable names follow the convention used on wikipedia.
    """
    with mp.extradps(5):
        a, b = _validate_ab(a, b)
        x = mp.mpf(x)
        if x < 1:
            return mp.ninf
        blogx = b*mp.log(x)
        c = (1 + 2*blogx/a)*(1 + a + 2*blogx) - 2*b/a
        return mp.log(c) - (2 + a + blogx)*mp.log(x)


def cdf(x, a, b):
    """
    CDF of the Benktander I distribution.

    Variable names follow the convention used on wikipedia.
    """
    with mp.extradps(5):
        a, b = _validate_ab(a, b)
        x = mp.mpf(x)
        if x < 1:
            return mp.zero
        blogx = b*mp.log(x)
        return 1 - (1 + 2*blogx/a)*mp.power(x, -(a + 1 + blogx))


def sf(x, a, b):
    """
    Survival function of the Benktander I distribution.

    Variable names follow the convention used on wikipedia.
    """
    with mp.extradps(5):
        a, b = _validate_ab(a, b)
        x = mp.mpf(x)
        if x < 1:
            return mp.one
        blogx = b*mp.log(x)
        return (1 + 2*blogx/a)*mp.power(x, -(a + 1 + blogx))


def invcdf(p, a, b):
    """
    Inverse of the CDF of the Benktander I distribution.

    Variable names follow the convention used on wikipedia.
    """
    with mp.extradps(5):
        p = _validate_p(p)
        a, b = _validate_ab(a, b)
        if p == 0:
            return mp.one
        if p == 1:
            return mp.inf
        w = mp.log1p(-p)
        zlow = (-(a + mp.one) + mp.sqrt((a + mp.one)**2 - 4*b*w)) / (2*b)
        q = a + mp.one - 2*b/a
        zhigh = (-q + mp.sqrt(q**2 - 4*b*w)) / (2*b)
        z = mp.findroot(lambda z: (mp.log(1 + 2*b/a*z)
                                   - (a + 1 + b*z)*z - w),
                        (zlow, zhigh), method='anderson')
        return mp.exp(z)


def invsf(p, a, b):
    """
    Inverse of the survival function of the Benktander I distribution.

    Variable names follow the convention used on wikipedia.
    """
    with mp.extradps(5):
        p = _validate_p(p)
        a, b = _validate_ab(a, b)
        if p == 0:
            return mp.inf
        if p == 1:
            return mp.one
        w = mp.log(p)
        zlow = (-(a + mp.one) + mp.sqrt((a + mp.one)**2 - 4*b*w)) / (2*b)
        q = a + mp.one - 2*b/a
        zhigh = (-q + mp.sqrt(q**2 - 4*b*w)) / (2*b)
        z = mp.findroot(lambda z: (mp.log(1 + 2*b/a*z)
                                   - (a + 1 + b*z)*z - w),
                        (zlow, zhigh), method='anderson')
        return mp.exp(z)


def support(a, b):
    """
    Support of the Benktander I distribution.
    """
    with mp.extradps(5):
        a, b = _validate_ab(a, b)
        return (mp.one, mp.inf)


def mean(a, b):
    """
    Mean of the Benktander I distribution.

    Variable names follow the convention used on wikipedia.
    """
    with mp.extradps(5):
        a, b = _validate_ab(a, b)
        return 1 + 1/a


def var(a, b):
    """
    Variance of the Benktander I distribution.

    Variable names follow the convention used on wikipedia.
    """
    with mp.extradps(5):
        a, b = _validate_ab(a, b)
        sb = mp.sqrt(b)
        t = (a - mp.one)/(2*sb)
        sqrtpi = mp.sqrt(mp.pi)
        return (-sb + a*mp.exp(t**2)*sqrtpi*mp.erfc(t))/(a**2*sb)


def nll(x, a, b):
    """
    Negative log-likelihood function for the Benktander I distribution.

    `x` must be a sequence of numbers with values greater than or equal
    to 1.
    """
    with mp.extradps(5):
        x = _validate_x_bounds(x, low=1)
        a, b = _validate_ab(a, b)
        return -mp.fsum([logpdf(t, a, b) for t in x])


def mle(x, *, a=None, b=None):
    """
    Maximum likelihood estimation for the Benktander I distribution.
    """
    if ((a is not None and not isinstance(a, Initial))
            or (b is not None and not isinstance(b, Initial))):
        raise ValueError('fixed parameters not implemented yet; a and b must '
                         'be None or Initial(val) for now')
    with mp.extradps(5):
        x = _validate_x_bounds(x, low=1)
        logx = [mp.log(t) for t in x]
        n = len(x)

        def mle_eqns(a, b):
            t1 = [2*a + 4*b*lnx + 1 for lnx in logx]
            t2 = [a**2 + 4*a*b*lnx + a + 4*b**2*lnx**2 + 2*b*lnx - 2*b
                  for lnx in logx]
            dlda = -n/a + mp.fsum((z1/z2) - lnx
                                  for z1, z2, lnx in zip(t1, t2, logx))
            dldb = mp.fsum((z1 * 2*lnx - 2)/z2 - lnx**2
                           for z1, z2, lnx in zip(t1, t2, logx))
            return dlda, dldb

        if isinstance(a, Initial):
            a0 = a.initial
        else:
            mu_hat = _mean(x)
            a0 = 1/(mu_hat - 1)
        b0 = b.initial if isinstance(b, Initial) else a0*(a0 + 1)/2
        a0, b0 = _validate_ab(a0, b0)
        a_hat, b_hat = mp.findroot(mle_eqns, [a0, b0])
        if a_hat <= 0:
            raise RuntimeError('findroot found invalid parameters: '
                               'a_hat is not positive')
        if b_hat <= 0:
            raise RuntimeError('findroot found invalid parameters: '
                               'b_hat is not positive')
        if b_hat > a_hat*(a_hat + 1)/2:
            raise RuntimeError('findroot found invalid parameters: '
                               'b_hat > a_hat*(a_hat + 1)/2')
        return a_hat, b_hat
