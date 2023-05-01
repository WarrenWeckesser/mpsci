from mpmath import mp
from ._xlogy import xlog1py


def inv_powm1(t, y):
    """
    Inverse with respect to x of ``mpmath.powm1(x, y)``.

    *See also:* :func:`pow1pm1`, :func:`inv_pow1pm1`
    """
    if y == 0:
        raise ValueError('y must not be 0.')
    if t <= -1:
        raise ValueError('t must be greater than -1.')
    with mp.extradps(5):
        t = mp.mpf(t)
        y = mp.mpf(y)
        return mp.exp(xlog1py(1/y, t))


def pow1pm1(x, y):
    """
    Compute ``(x + 1)**y - 1``.

    *See also:* :func:`inv_pow1pm1`
    """
    with mp.extradps(5):
        x = mp.mpf(x)
        y = mp.mpf(y)
        return mp.expm1(xlog1py(y, x))


def inv_pow1pm1(t, y):
    """
    Inverse with respect to x of ``pow1pm1(x, y)``.

    *See also:* :func:`pow1pm1`
    """
    if y == 0:
        raise ValueError('y must not be 0.')
    if t <= -1:
        raise ValueError('t must be greater than -1.')
    with mp.extradps(5):
        t = mp.mpf(t)
        y = mp.mpf(y)
        return mp.expm1(xlog1py(1/y, t))
