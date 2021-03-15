"""
Benktander I Distribution
-------------------------
"""

import mpmath


__all__ = ['pdf', 'logpdf', 'cdf', 'sf', 'invcdf', 'invsf', 'mean', 'var']


def _validate_ab(a, b):
    if a <= 0:
        raise ValueError("'a' must be positive.")
    if b <= 0:
        raise ValueError("'b' must be positive.")
    if b > a*(a + 1)/2:
        raise ValueError("'b' must not be greater than a*(a+1)/2.")


def pdf(x, a, b):
    """
    PDF of the Benktander I distribution.

    Variable names follow the convention used on wikipedia.
    """
    _validate_ab(a, b)
    if x < 1:
        return mpmath.mp.zero
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        a = mpmath.mpf(a)
        b = mpmath.mpf(b)
        blogx = b*mpmath.log(x)
        c = (1 + 2*blogx/a)*(1 + a + 2*blogx) - 2*b/a
        return c * mpmath.power(x, -(2 + a + blogx))


def logpdf(x, a, b):
    """
    Logarithm of the PDF of the Benktander I distribution.

    Variable names follow the convention used on wikipedia.
    """
    _validate_ab(a, b)
    if x < 1:
        return mpmath.ninf
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        a = mpmath.mpf(a)
        b = mpmath.mpf(b)
        blogx = b*mpmath.log(x)
        c = (1 + 2*blogx/a)*(1 + a + 2*blogx) - 2*b/a
        return mpmath.log(c) - (2 + a + blogx)*mpmath.log(x)


def cdf(x, a, b):
    """
    CDF of the Benktander I distribution.

    Variable names follow the convention used on wikipedia.
    """
    _validate_ab(a, b)
    if x < 1:
        return mpmath.mp.zero
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        a = mpmath.mpf(a)
        b = mpmath.mpf(b)
        blogx = b*mpmath.log(x)
        return 1 - (1 + 2*blogx/a)*mpmath.power(x, -(a + 1 + blogx))


def sf(x, a, b):
    """
    Survival function of the Benktander I distribution.

    Variable names follow the convention used on wikipedia.
    """
    _validate_ab(a, b)
    if x < 1:
        return mpmath.mp.one
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        a = mpmath.mpf(a)
        b = mpmath.mpf(b)
        blogx = b*mpmath.log(x)
        return (1 + 2*blogx/a)*mpmath.power(x, -(a + 1 + blogx))


def invcdf(p, a, b):
    """
    Inverse of the CDF of the Benktander I distribution.

    Variable names follow the convention used on wikipedia.
    """
    _validate_ab(a, b)
    with mpmath.extradps(5):
        p = mpmath.mpf(p)
        a = mpmath.mpf(a)
        b = mpmath.mpf(b)
        one = mpmath.mp.one
        w = mpmath.log1p(-p)
        zlow = (-(a + one) + mpmath.sqrt((a + one)**2 - 4*b*w)) / (2*b)
        q = a + one - 2*b/a
        zhigh = (-q + mpmath.sqrt(q**2 - 4*b*w)) / (2*b)
        z = mpmath.findroot(lambda z: (mpmath.log(1 + 2*b/a*z)
                                       - (a + 1 + b*z)*z - w),
                            (zlow, zhigh), method='anderson')
        return mpmath.exp(z)


def invsf(p, a, b):
    """
    Inverse of the survival function of the Benktander I distribution.

    Variable names follow the convention used on wikipedia.
    """
    _validate_ab(a, b)
    with mpmath.extradps(5):
        p = mpmath.mpf(p)
        a = mpmath.mpf(a)
        b = mpmath.mpf(b)
        one = mpmath.mp.one
        w = mpmath.log(p)
        zlow = (-(a + one) + mpmath.sqrt((a + one)**2 - 4*b*w)) / (2*b)
        q = a + one - 2*b/a
        zhigh = (-q + mpmath.sqrt(q**2 - 4*b*w)) / (2*b)
        z = mpmath.findroot(lambda z: (mpmath.log(1 + 2*b/a*z)
                                       - (a + 1 + b*z)*z - w),
                            (zlow, zhigh), method='anderson')
        return mpmath.exp(z)


def mean(a, b):
    """
    Mean of the Benktander I distribution.

    Variable names follow the convention used on wikipedia.
    """
    _validate_ab(a, b)
    with mpmath.extradps(5):
        a = mpmath.mpf(a)
        return 1 + 1/a


def var(a, b):
    """
    Variance of the Benktander I distribution.

    Variable names follow the convention used on wikipedia.
    """
    _validate_ab(a, b)
    with mpmath.extradps(5):
        a = mpmath.mpf(a)
        b = mpmath.mpf(b)
        sb = mpmath.sqrt(b)
        t = (a - mpmath.mp.one)/(2*sb)
        sqrtpi = mpmath.sqrt(mpmath.pi)
        return (-sb + a*mpmath.exp(t**2)*sqrtpi*mpmath.erfc(t))/(a**2*sb)
