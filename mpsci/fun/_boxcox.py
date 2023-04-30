
from mpmath import mp
import re
from ._powm1 import pow1pm1, inv_powm1, inv_pow1pm1

__all__ = ['boxcox', 'boxcox1p']


def boxcox(x, lmbda):
    r"""
    Box-Cox transformation of x.

    The Box-Cox transformation is

                      { log(x)          if lmbda == 0,
        f(x; lmbda) = {
                      { x**lmbda - 1
                      { ------------    if lmbda != 0
                      {    lmbda

    *See also:* :func:`inv_boxcox`,  :func:`boxcox1p`
    """
    x = mp.mpf(x)
    lmbda = mp.mpf(lmbda)
    if lmbda == 0:
        return mp.log(x)
    else:
        return mp.powm1(x, lmbda) / lmbda


_boxcox_latex_f = r"""
.. math::

        f(x; \\lambda) = \\begin{cases}
                           \\log(x) & \\textrm{if} \\; \\lambda = 0 \\\\
                           \\frac{x^{\\lambda} - 1}{\\lambda}  & \\textrm{if} \\; \\lambda \\ne 0
                        \\end{cases}
"""

boxcox._docstring_re_subs = [
    ('[ ]+{.*lmbda', _boxcox_latex_f, 0, re.DOTALL)
]


def boxcox1p(x, lmbda):
    r"""
    Box-Cox transformation of 1 + x.

    The transformation is

                      { log(1+x)            if lmbda == 0,
        f(x; lmbda) = {
                      { (1+x)**lmbda - 1
                      { ----------------    if lmbda != 0
                      {      lmbda

    This function is mathematically equivalent to `boxcox(1+x, lmbda)`.
    It avoids the loss of precision that can occur if x is very small.

    *See also:* :func:`boxcox`, :func:`inv_boxcox1p`
    """
    x = mp.mpf(x)
    lmbda = mp.mpf(lmbda)
    if lmbda == 0:
        return mp.log1p(x)
    else:
        return pow1pm1(x, lmbda) / lmbda


_boxcox1p_latex_f = r"""
.. math::

      f(x; \\lambda) = \\begin{cases}
                         \\log(x + 1) & \\textrm{if} \\; \\lambda = 0 \\\\
                         \\frac{(x + 1)^{\\lambda} - 1}{\\lambda}  & \\textrm{if} \\; \\lambda \\ne 0
                      \\end{cases}
"""

boxcox1p._docstring_re_subs = [
    ('[ ]+{.*    lmbda', _boxcox1p_latex_f, 0, re.DOTALL)
]


def inv_boxcox(y, lmbda):
    """
    Inverse with respect to x of boxcox(x, lmbda).

    *See also:* :func:`boxcox`
    """
    y = mp.mpf(y)
    lmbda = mp.mpf(lmbda)
    if lmbda == 0:
        return mp.exp(y)
    else:
        return inv_powm1(lmbda*y, lmbda)


def inv_boxcox1p(y, lmbda):
    """
    Inverse with respect to x of boxcox1p(x, lmbda).

    *See also:* :func:`boxcox1p`
    """
    y = mp.mpf(y)
    lmbda = mp.mpf(lmbda)
    if lmbda == 0:
        return mp.expm1(y)
    else:
        return inv_pow1pm1(lmbda*y, lmbda)
