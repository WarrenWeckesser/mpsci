from mpmath import mp


__all__ = ['digammainv']


def digammainv(y):
    """
    Inverse of the digamma function (real values only).

    The `digamma function` [1]_ [2]_ is the logarithmic derivative of the
    gamma function.

    For real ``y``, ``digammainv(y)`` returns the positive ``x`` such that
    ``digamma(x) = y``.

    The digamma function is also known as psi_0; ``mpmath.digamma(x)`` is the
    same as ``mpmath.psi(0, x)``.

    References
    ----------
    .. [1] "Digamma function",
           https://en.wikipedia.org/wiki/Digamma_function
    .. [2] Abramowitz and Stegun, *Handbook of Mathematical Functions*
           (section 6.3), Dover Publications, New York (1972).

    Examples
    --------

    >>> from mpmath import mp
    >>> from mpsci.fun import digammainv
    >>> mp.dps = 25
    >>> y = mpmath.mpf('7.89123')
    >>> y
    mpf('7.891230000000000000000000011')
    >>> x = digammainv(y)
    >>> x
    mpf('2674.230572001301673812839151')
    >>> mp.digamma(x)
    mpf('7.891230000000000000000000011')
    """

    with mp.extradps(5):
        y = mp.mpf(y)

        # Find a good initial guess for the root.
        if y > -0.125:
            x0 = mp.exp(y) + mp.mpf('0.5')
        elif y > -3:
            x0 = mp.exp(y/mp.mpf(2.332)) + mp.mpf(0.08661)
        else:
            x0 = -1 / (y + mp.euler)

        solver = 'anderson'
        x0 = (4*x0/5, 5*x0/4)
        x = mp.findroot(lambda x: mp.digamma(x) - y, x0, solver=solver)

    return x


digammainv._docstring_re_subs = [
    ('psi_0', r':math:`\\psi_{0}`', 0, 0)
]
