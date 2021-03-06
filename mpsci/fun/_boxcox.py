
import mpmath
import re

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

    *See also:* `mpsci.fun.boxcox1p`
    """
    x = mpmath.mpf(x)
    lmbda = mpmath.mpf(lmbda)
    if lmbda == 0:
        return mpmath.log(x)
    else:
        return mpmath.powm1(x, lmbda) / lmbda


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

    This function is mathematically equivalent to `boxcox(1+x, lmba)`.
    It avoids the loss of precision that can occur if x is very small.

    *See also:* `mpsci.fun.boxcox`
    """
    x = mpmath.mpf(x)
    lmbda = mpmath.mpf(lmbda)
    one = mpmath.mpf(1)
    if lmbda == 0:
        return mpmath.log1p(x)
    else:
        return mpmath.powm1(one + x, lmbda) / lmbda


_boxcox1p_latex_f = r"""
.. math::

      f(x; \\lambda) = \\begin{cases}
                         \\log(x + 1) & \\textrm{if} \\; \\lambda = 0 \\\\
                         \\frac{(x + 1)^{\\lambda} - 1}{\\lambda}  & \\textrm{if} \\; \\lambda \\ne 0
                      \\end{cases}
"""

boxcox1p._docstring_re_subs = [
    ('[ ]+{.*lmbda', _boxcox1p_latex_f, 0, re.DOTALL)
]
