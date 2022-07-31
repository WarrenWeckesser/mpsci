
import mpmath


def linspace(a, b, n):
    """
    Return a list of ``n`` evenly spaced numbers starting at ``a`` and
    ending at ``b``.

    ``n`` must be an integer, and must be at least 2.

    Examples
    --------
    >>> import mpmath
    >>> mpmath.mp.dps = 25

    >>> from mpsci.func import linspace

    >>> linspace(0, 1, 9)
    [mpf('0.0'),
     mpf('0.125'),
     mpf('0.25'),
     mpf('0.375'),
     mpf('0.5'),
     mpf('0.625'),
     mpf('0.75'),
     mpf('0.875'),
     mpf('1.0')]

    >>> linspace(-10, 10, 13)
    [mpf('-10.0'),
     mpf('-8.333333333333333333333333333'),
     mpf('-6.666666666666666666666666667'),
     mpf('-5.0'),
     mpf('-3.333333333333333333333333333'),
     mpf('-1.666666666666666666666666667'),
     mpf('0.0'),
     mpf('1.666666666666666666666666667'),
     mpf('3.333333333333333333333333333'),
     mpf('5.0'),
     mpf('6.666666666666666666666666667'),
     mpf('8.333333333333333333333333333'),
     mpf('10.0')]
    """
    m = int(n)
    if m != n:
        raise ValueError('n is not an integer')
    if m < 2:
        raise ValueError('n must be at least 2')

    with mpmath.extradps(5):
        a = mpmath.mpf(a)
        b = mpmath.mpf(b)
        delta = (b - a)/(m - 1)
        result = [a + k*delta for k in range(m)]
        result[-1] = b
        return result
