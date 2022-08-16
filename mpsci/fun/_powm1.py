
import mpmath
from ._xlogy import xlog1py


def inv_powm1(t, y):
    """
    Inverse with respect to x of powm1(x, y).
    """
    if y == 0:
        raise ValueError('y must not be 0.')
    if t <= -1:
        raise ValueError('t must be greater than -1.')
    with mpmath.extradps(5):
        t = mpmath.mpf(t)
        y = mpmath.mpf(y)
        return mpmath.exp(xlog1py(1/y, t))


def pow1pm1(x, y):
    """
    Compute (x + 1)**y - 1.
    """
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        y = mpmath.mpf(y)
        return mpmath.expm1(xlog1py(y, x))


def inv_pow1pm1(t, y):
    """
    Inverse with respect to x of pow1pm1(x, y).
    """
    if y == 0:
        raise ValueError('y must not be 0.')
    if t <= -1:
        raise ValueError('t must be greater than -1.')
    with mpmath.extradps(5):
        t = mpmath.mpf(t)
        y = mpmath.mpf(y)
        return mpmath.expm1(xlog1py(1/y, t))
