"""
Benktander I Distribution
-------------------------
"""

from mpmath import mp
from ._common import _validate_p


__all__ = ['pdf', 'logpdf', 'cdf', 'sf', 'invcdf', 'invsf', 'mean', 'var']


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
        w = mp.log(p)
        zlow = (-(a + mp.one) + mp.sqrt((a + mp.one)**2 - 4*b*w)) / (2*b)
        q = a + mp.one - 2*b/a
        zhigh = (-q + mp.sqrt(q**2 - 4*b*w)) / (2*b)
        z = mp.findroot(lambda z: (mp.log(1 + 2*b/a*z)
                                   - (a + 1 + b*z)*z - w),
                        (zlow, zhigh), method='anderson')
        return mp.exp(z)


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
