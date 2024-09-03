import re
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
    Compute (1 + x)**y - 1.

    *See also:* :func:`inv_pow1pm1`

    Examples
    --------
    >>> from mpsci.fun import pow1pm1
    >>> from mpmath import mp
    >>> mp.dps = 25

    >>> x = mp.mpf('3.5e-75')
    >>> y = mp.mpf('2.86e-8')

    Naive calculation:

    >>> (1 + x)**y - 1
    mpf('0.0')

    Using ``pow1pm1``:

    >>> pow1pm1(x, y)
    mpf('1.001000000000000000000000001e-82')

    """
    with mp.extradps(5):
        x = mp.mpf(x)
        y = mp.mpf(y)
        return mp.expm1(xlog1py(y, x))


pow1pm1._docstring_re_subs = [
    (r'Compute \(1.*- 1', r'Compute :math:`(1 + x)^y - 1`', 0, 0)
]


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
