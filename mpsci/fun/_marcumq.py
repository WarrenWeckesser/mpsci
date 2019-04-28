
import re
import mpmath


__all__ = ['marcumq', 'cmarcumq']


def _integrand(x, m, a):
    e2 = mpmath.exp(-(x**2 + a**2)/2)
    return x*(x/a)**(m - 1)*e2*mpmath.besseli(m-1, a*x)


def marcumq(m, a, b):
    """
    The Marcum Q function.

    The function is defined as the integral from b to infinity of

        x*(x/a)**(M-1)*exp(-(x**2 + a**2)/2)*I_{M-1}(a*x)

    where I_{M-1}(x) is the modified Bessel function of order M-1.

    *See also:* `mpsci.fun.cmarcumq`

    Returns
    -------
    q : mpmath.mpf
        The function value.

    Notes
    -----
    The function uses numerical integration, so it can be very slow.

    Examples
    --------
    >>> import mpmath
    >>> from mpsci.fun import marcumq
    >>> mpmath.mp.dps = 40
    >>> marcumq(2, 0.5, 3.0)
    mpf('0.07558718754263240906718640000640082605610073')

    """
    if a == 0:
        if m == 1:
            q = mpmath.exp(-b**2/2)
        else:
            q = mpmath.gammainc(m, b**2/2, regularized=True)
    elif b == 0 and m > 0:
        q = mpmath.mpf(1)
    else:
        q = mpmath.quad(lambda x: _integrand(x, m, a), [b, mpmath.inf])
    return q

marcumq._docstring_re_subs = [
    ('is defined as .*a\*x\)',
     r'''is defined as

.. math::

   Q_M(a, b) = \\int_b^{\\infty} x \\left(\\frac{x}{a}\\right)^{M-1}
               \\exp\\left(-\\frac{x^2 + a^2}{2}\\right) I_{M-1}(ax)\\,dx
''', 0, re.DOTALL),
    ('I_\{M-1\}\(x\)', r':math:`I_{M-1}(z)`', 0, 0),
    ('M-1\.', r':math:`M-1`', 0, 0),
]

def cmarcumq(m, a, b):
    """
    The "complementary" Marcum Q function.

    This is 1 - marcumq(m, a, b).

    The function uses numerical integration, so it can be very slow.

    The computed integral tends to be very inaccurate for m < 1/2.

    *See also:* `mpsci.fun.marcumq`
    """
    if a == 0:
        if m == 1:
            q = -mpmath.expm1(-b**2/2)
        else:
            q = mpmath.gammainc(m, 0, b**2/2, regularized=True)
    elif b == 0 and m > 0:
        q = mpmath.mpf(0)
    else:
        q = mpmath.quad(lambda x: _integrand(x, m, a), [0, b])
    return q
