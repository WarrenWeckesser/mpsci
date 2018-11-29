
import mpmath


def pdf(x, a, c):
    """
    PDF for the exponentiated Weibull distribution.
    """
    x = mpmath.mpf(x)
    a = mpmath.mpf(a)
    c = mpmath.mpf(c)
    p = a * c * x**(c-1) * (- mpmath.expm1(-x**c))**(a - 1) * mpmath.exp(-x**c)
    return p
