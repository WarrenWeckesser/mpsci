"""
Benktander II Distribution
--------------------------
"""

import mpmath


__all__ = ['pdf', 'logpdf', 'cdf', 'sf', 'invcdf', 'invsf', 'mean', 'var']


def _validate_ab(a, b):
    if a <= 0:
        raise ValueError("'a' must be positive.")
    if b <= 0 or b > 1:
        raise ValueError("'b' must be in the interval (0, 1].")


def pdf(x, a, b):
    """
    PDF of the Benktander II distribution.

    Variable names follow the convention used on wikipedia.
    """
    _validate_ab(a, b)
    if x < 1:
        return mpmath.mp.zero
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        a = mpmath.mpf(a)
        b = mpmath.mpf(b)
        p1 = mpmath.exp((a/b)*(1 - x**b))
        p2 = x**(b - 2)
        p3 = (a*x**b - b + 1)
        return p1*p2*p3


def logpdf(x, a, b):
    """
    Logarithm of the PDF of the Benktander II distribution.

    Variable names follow the convention used on wikipedia.
    """
    _validate_ab(a, b)
    if x < 1:
        return mpmath.ninf
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        a = mpmath.mpf(a)
        b = mpmath.mpf(b)
        t1 = (a/b)*(1 - x**b)
        t2 = (b - 2)*mpmath.log(x)
        t3 = mpmath.log1p(a*x**b - b)
        return t1 + t2 + t3


def cdf(x, a, b):
    """
    CDF of the Benktander II distribution.

    Variable names follow the convention used on wikipedia.
    """
    _validate_ab(a, b)
    if x < 1:
        return mpmath.mp.zero
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        a = mpmath.mpf(a)
        b = mpmath.mpf(b)
        return 1 - x**(b - 1)*mpmath.exp((a/b)*(1 - x**b))


def sf(x, a, b):
    """
    Survival function of the Benktander II distribution.

    Variable names follow the convention used on wikipedia.
    """
    _validate_ab(a, b)
    if x < 1:
        return mpmath.mp.one
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        a = mpmath.mpf(a)
        b = mpmath.mpf(b)
        return x**(b - 1)*mpmath.exp((a/b)*(1 - x**b))


def invcdf(p, a, b):
    """
    Inverse CDF of the Benktander II distribution.

    Variable names follow the convention used on wikipedia.
    """
    _validate_ab(a, b)
    with mpmath.extradps(5):
        p = mpmath.mpf(p)
        a = mpmath.mpf(a)
        b = mpmath.mpf(b)
        w = mpmath.log1p(-p)
        one = mpmath.mp.one
        # Compute a bracketing interval for the root.
        xlow = one + w/(b - one - a)
        xhigh = mpmath.power(one - (b/a)*w, 1/b)

        def func(x):
            return w - (b - 1)*mpmath.log(x) - (a/b)*(one - x**b)

        x = mpmath.findroot(func, (xlow, xhigh), method='anderson')
        return x


def invsf(p, a, b):
    """
    Inverse survival function of the Benktander II distribution.

    Variable names follow the convention used on wikipedia.
    """
    _validate_ab(a, b)
    with mpmath.extradps(5):
        p = mpmath.mpf(p)
        a = mpmath.mpf(a)
        b = mpmath.mpf(b)
        w = mpmath.log(p)
        one = mpmath.mp.one
        # Compute a bracketing interval for the root.
        xlow = one + w/(b - one - a)
        xhigh = mpmath.power(one - (b/a)*w, 1/b)

        def func(x):
            return w - (b - 1)*mpmath.log(x) - (a/b)*(one - x**b)

        x = mpmath.findroot(func, (xlow, xhigh), method='anderson')
        return x


def mean(a, b):
    """
    Mean of the Benktander II distribution.

    Variable names follow the convention used on wikipedia.
    """
    _validate_ab(a, b)
    with mpmath.extradps(5):
        a = mpmath.mpf(a)
        return 1 + 1/a


def var(a, b):
    """
    Variance of the Benktander II distribution.

    Variable names follow the convention used on wikipedia.
    """
    _validate_ab(a, b)
    with mpmath.extradps(5):
        a = mpmath.mpf(a)
        b = mpmath.mpf(b)
        r = a/b
        return (-b + 2*a*mpmath.exp(r)*mpmath.expint(1 - 1/b, r))/(a**2*b)
