"""
Raised cosine distribution
--------------------------

"""

import re
import mpmath


__all__ = ['pdf', 'cdf', 'invcdf']


# TO DO: Add location and scale parameters.


def pdf(x):
    """
    Probability density function (PDF) of the raised cosine distribution.

    The PDF of the raised cosine distribution is

        f(x) = (1 + cos(x))/(2*pi)

    on the interval (-pi, pi) and zero elsewhere.
    """
    with mpmath.extradps(5):
        return (mpmath.cos(x) + 1)/(2*mpmath.pi)


_pdf_docstring_replace = r"""
    .. math::

       f(x) = \\frac{1 + \\cos(x)}{2\\pi}

"""

pdf._docstring_re_subs = [
    ('    f.*\*pi\)', _pdf_docstring_replace, 0, re.DOTALL),
    ('\(-pi, pi\)', r':math:`(-\\pi, \\pi)`', 0, 0),
]

def cdf(x):
    """
    Cumulative distribution function (CDF) of the raised cosine distribution.

    The CDF of the raised cosine distribution is

        F(x) = (pi + x + sin(x))/(2*pi)
    """
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        return mpmath.mpf('1/2') + (x + mpmath.sin(x))/(2*mpmath.pi)


_cdf_docstring_replace = r"""
    .. math::

       F(x) = \\frac{\\pi + x + \\sin(x)}{2\\pi}

"""

cdf._docstring_re_subs = [
    ('    F.*pi\)', _cdf_docstring_replace, 0, re.DOTALL),
]


def invcdf(p):
    """
    Inverse of the CDF of the raised cosine distribution.

    XXX This implementation needs further testing, especially for the
    behavior near x=pi and x=-pi.
    """

    if p < 0 or p > 1:
        return mpmath.nan
    if p == 0:
        return -mpmath.pi
    if p == 1:
        return mpmath.pi

    def rootfunc(t):
        return cdf(t) - p

    with mpmath.extradps(5):
        xp1 = mpmath.mpf('-1.5148393083566466647517666551356551803961998987290333367840767')
        xp9 = mpmath.mpf('1.51483930835664682990834405453173140711667691817143014741422715')

        p = mpmath.mpf(p)

        solver = 'bisect'
        if p < 0.09:
            x0 = (-mpmath.pi, xp1)
            x = mpmath.findroot(rootfunc, x0, solver=solver)
        elif p > 0.91:
            x0 = (xp9, mpmath.pi)
            x = mpmath.findroot(rootfunc, x0, solver=solver)
        else:
            # 0.1 <= p <= 0.9
            x0 = 0.0
            x = mpmath.findroot(rootfunc, x0)
        return x
