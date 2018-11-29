
import mpmath


__all__ = ['marcumq']


def _integrand(x, m, a, b):
    e2 = mpmath.exp(-(x**2 + a**2)/2)
    return x*(x/a)**(m - 1)*e2*mpmath.besseli(m-1, a*x)


def marcumq(m, a, b):
    """
    The Marcum Q function.

    The currently implementation uses numerical integration, so the function
    can be very slow.
    """
    q = mpmath.quad(lambda x: _integrand(x, m, a, b), [b, mpmath.inf])
    return q
