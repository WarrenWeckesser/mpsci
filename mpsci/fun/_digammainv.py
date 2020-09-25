
import mpmath


__all__ = ['digammainv']


def digammainv(y):
    """
    Inverse of the digamma function (real values only).

    The `digamma function` [1]_ [2]_ is the logarithmic derivative of the
    gamma function.

    For real y, digammainv(y) returns the positive x such that digamma(x) = y.

    The digamma function is also known as psi_0; `mpmath.digamma(x)` is the
    same as `mpmath.psi(0, x)`.

    References
    ----------
    .. [1] "Digamma function",
           https://en.wikipedia.org/wiki/Digamma_function
    .. [2] Abramowitz and Stegun, *Handbook of Mathematical Functions*
           (section 6.3), Dover Publications, New York (1972).

    Examples
    --------

    >>> import mpmath
    >>> from mpsci.fun import digammainv
    >>> mpmath.mp.dps = 25
    >>> y = mpmath.mpf('7.89123')
    >>> y
    mpf('7.891230000000000000000000011')
    >>> x = digammainv(y)
    >>> x
    mpf('2674.230572001301673812839151')
    >>> mpmath.digamma(x)
    mpf('7.891230000000000000000000011')
    """

    # XXX I'm not sure this extra dps is necessary.
    with mpmath.extradps(5):
        y = mpmath.mpf(y)

        # Find a good initial guess for the root.
        if y > -0.125:
            x0 = mpmath.exp(y) + mpmath.mpf('0.5')
        elif y > -3:
            x0 = mpmath.exp(y/mpmath.mpf(2.332)) + mpmath.mpf(0.08661)
        else:
            x0 = -1 / (y + mpmath.euler)

        solver = 'anderson'
        x0 = (4*x0/5, 5*x0/4)
        x = mpmath.findroot(lambda x: mpmath.digamma(x) - y, x0, solver=solver)

    return x


digammainv._docstring_re_subs = [
    ('psi_0', r':math:`\\psi_{0}`', 0, 0)
]
