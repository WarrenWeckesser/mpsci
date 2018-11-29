
import mpmath


__all__ = ['marcumq', 'cmarcumq']


def _integrand(x, m, a):
    e2 = mpmath.exp(-(x**2 + a**2)/2)
    return x*(x/a)**(m - 1)*e2*mpmath.besseli(m-1, a*x)


def marcumq(m, a, b):
    """
    The Marcum Q function.

    The function uses numerical integration, so it can be very slow.
    """
    q = mpmath.quad(lambda x: _integrand(x, m, a), [b, mpmath.inf])
    return q


def cmarcumq(m, a, b):
    """
    The "complementary" Marcum Q function.

    This is 1 - marcumq(m, a, b).

    The function uses numerical integration, so it can be very slow.
    """
    q = mpmath.quad(lambda x: _integrand(x, m, a), [0, b])
    return q
