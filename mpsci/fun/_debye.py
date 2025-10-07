import operator
from mpmath import mp


def debye(x, *, n, method='quad'):
    """
    Compute the Debye function D_n(x).

    If ``method == 'quad'``,  ``mp.quad`` is used to evaluate the integral form
    of the function, so it might be necessary to set ``mp.dps`` to a value
    much larger than the desired output precision.  For example, ``mp.dps = 150``
    is required to get an accurate double precision result from
    ``mp_debye1(1e100, 1)``.

    If ``method == 'nsum'``, ``mp.nsum`` is used to evalute the series form
    of the function.
    """
    try:
        n = operator.index(n)
    except TypeError:
        raise ValueError('n must be an integer')
    if n < 1:
        raise ValueError('n must be an integer greater than 0')

    if method not in ['quad', 'nsum']:
        raise ValueError("method must be 'quad' or 'nsum'")

    x = mp.mpf(x)

    if x == 0:
        return mp.one

    if method == 'quad':

        def integrand(t):
            if t == 0:
                if n == 0:
                    return mp.one
                return mp.zero
            return t**n / mp.expm1(t)

        return n*mp.quad(integrand, [0, x])/x**n

    # method == 'nsum'

    def term(k):
        twok = 2*k
        return mp.bernoulli(twok)/(twok + n)/mp.factorial(twok)*x**twok

    return 1 - n*x/(2*(n + 1)) + n*mp.nsum(term, [1, mp.inf])
