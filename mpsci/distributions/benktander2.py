"""
Benktander II Distribution
--------------------------
"""

from mpmath import mp
from ._common import _validate_p


__all__ = ['pdf', 'logpdf', 'cdf', 'sf', 'invcdf', 'invsf',
           'support', 'mean', 'var']


def _validate_ab(a, b):
    if a <= 0:
        raise ValueError("'a' must be positive.")
    if b <= 0 or b > 1:
        raise ValueError("'b' must be in the interval (0, 1].")
    return mp.mpf(a), mp.mpf(b)


def pdf(x, a, b):
    """
    PDF of the Benktander II distribution.

    Variable names follow the convention used on wikipedia.
    """
    with mp.extradps(5):
        a, b = _validate_ab(a, b)
        x = mp.mpf(x)
        if x < 1:
            return mp.zero
        p1 = mp.exp((a/b) * -mp.powm1(x, b))
        p2 = x**(b - 2)
        p3 = (a*x**b - b + 1)
        return p1 * p2 * p3


def logpdf(x, a, b):
    """
    Logarithm of the PDF of the Benktander II distribution.

    Variable names follow the convention used on wikipedia.
    """
    with mp.extradps(5):
        a, b = _validate_ab(a, b)
        x = mp.mpf(x)
        if x < 1:
            return mp.ninf
        t1 = (a/b) * -mp.powm1(x, b)
        t2 = (b - 2)*mp.log(x)
        t3 = mp.log1p(a*x**b - b)
        return t1 + t2 + t3


def cdf(x, a, b):
    """
    CDF of the Benktander II distribution.

    Variable names follow the convention used on wikipedia.
    """
    with mp.extradps(5):
        a, b = _validate_ab(a, b)
        x = mp.mpf(x)
        if x < 1:
            return mp.zero
        p = (b - 1)*mp.log(x) + a/b*-mp.powm1(x, b)
        return -mp.expm1(p)


def sf(x, a, b):
    """
    Survival function of the Benktander II distribution.

    Variable names follow the convention used on wikipedia.
    """
    with mp.extradps(5):
        a, b = _validate_ab(a, b)
        x = mp.mpf(x)
        if x < 1:
            return mp.one
        return x**(b - 1)*mp.exp((a/b) * -mp.powm1(x, b))


def invcdf(p, a, b):
    """
    Inverse CDF of the Benktander II distribution.

    Variable names follow the convention used on wikipedia.
    """
    with mp.extradps(5):
        p = _validate_p(p)
        a, b = _validate_ab(a, b)
        if b == 1:
            return mp.one - mp.log1p(-p)/a
        else:
            onemb = mp.one - b
            c = a/onemb
            t = c*mp.exp(c)*mp.power(mp.one - p, -b/onemb)
            return mp.power(mp.lambertw(t)/c, 1/b)


def invsf(p, a, b):
    """
    Inverse survival function of the Benktander II distribution.

    Variable names follow the convention used on wikipedia.
    """
    with mp.extradps(5):
        p = _validate_p(p)
        a, b = _validate_ab(a, b)
        if b == 1:
            return mp.one - mp.log(p)/a
        else:
            onemb = mp.one - b
            c = a/onemb
            t = c*mp.exp(c)*mp.power(p, -b/onemb)
            return mp.power(mp.lambertw(t)/c, 1/b)


def support(a, b):
    """
    Support of the Benktander II distribution.
    """
    with mp.extradps(5):
        a, b = _validate_ab(a, b)
        return (mp.one, mp.inf)


def mean(a, b):
    """
    Mean of the Benktander II distribution.

    Variable names follow the convention used on wikipedia.
    """
    with mp.extradps(5):
        a, b = _validate_ab(a, b)
        return 1 + 1/a


def var(a, b):
    """
    Variance of the Benktander II distribution.

    Variable names follow the convention used on wikipedia.
    """
    with mp.extradps(5):
        a, b = _validate_ab(a, b)
        r = a/b
        return (-b + 2*a*mp.exp(r)*mp.expint(1 - 1/b, r))/(a**2*b)
