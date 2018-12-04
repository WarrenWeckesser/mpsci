
import mpmath


__all__ = ['digammainv']


def digammainv(y):
    """
    Inverse of the digamma function (real values only).

    For real y, digammainv(y) returns x such that digamma(x) = y.

    The digamma function is also known as psi_0; `mpmath.digamma(x)` is the
    same as `mpmath.psi(0, x)`.
    """
    y = mpmath.mpf(y)

    # Find a good initial guess for the root.
    if y > -0.125:
        x0 = mpmath.exp(y) + 0.5
    elif y > -3:
        x0 = mpmath.exp(y/mpmath.mpf(2.332)) + mpmath.mpf(0.08661)
    else:
        x0 = 1 / (-y - mpmath.euler)

    solver = 'anderson'
    x0 = (4*x0/5, 5*x0/4)
    x = mpmath.findroot(lambda x: mpmath.digamma(x) - y, x0, solver=solver)

    return x
