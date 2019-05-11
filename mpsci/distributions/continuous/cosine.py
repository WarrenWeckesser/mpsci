"""
Raised cosine distribution
--------------------------

"""

import re
import mpmath


__all__ = ['pdf', 'logpdf', 'cdf', 'invcdf']


# TO DO: Add location and scale parameters.


def pdf(x):
    """
    Probability density function (PDF) of the raised cosine distribution.

    The PDF of the raised cosine distribution is

        f(x) = (1 + cos(x))/(2*pi)

    on the interval (-pi, pi) and zero elsewhere.
    """
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        if x <= -mpmath.pi or x >= mpmath.pi:
            return mpmath.mp.zero
        return (mpmath.cos(x) + 1)/(2*mpmath.pi)


_pdf_docstring_replace = r"""
    .. math::

       f(x) = \\frac{1 + \\cos(x)}{2\\pi}

"""

pdf._docstring_re_subs = [
    ('    f.*\*pi\)', _pdf_docstring_replace, 0, re.DOTALL),
    ('\(-pi, pi\)', r':math:`(-\\pi, \\pi)`', 0, 0),
]


def logpdf(x):
    """
    Natual logarithm of the PDF of the raised cosine distribution.

    The PDF of the raised cosine distribution is

        f(x) = (1 + cos(x))/(2*pi)

    on the interval (-pi, pi) and zero elsewhere.
    """
    with mpmath.extradps(5):
        if x <= -mpmath.pi or x >= mpmath.pi:
            return -mpmath.inf
        return mpmath.log1p(mpmath.cos(x)) - mpmath.log(2*mpmath.pi)


logpdf._docstring_re_subs = [
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
        if x <= -mpmath.pi:
            return mpmath.mp.zero
        if x >= mpmath.pi:
            return mpmath.mp.one
        return mpmath.mpf('1/2') + (x + mpmath.sin(x))/(2*mpmath.pi)


_cdf_docstring_replace = r"""
    .. math::

       F(x) = \\frac{\\pi + x + \\sin(x)}{2\\pi}

"""

cdf._docstring_re_subs = [
    ('    F.*pi\)', _cdf_docstring_replace, 0, re.DOTALL),
]


# XXX coefficients in _p2 and _q2 are 64 bit floating point, not
# mpmath floats.

def _p2(t):
    t = mpmath.mpf(t)
    return (0.5
            + t*(-0.06532856457583547
                 + t*(0.0020893844847965047
                      + t*-1.0233693819385904e-05)))

def _q2(t):
    t = mpmath.mpf(t)
    return (1.0
            + t*(-0.15149046248500425
                 + t*(0.006293153604697265
                      + t*-6.042645518776793e-05)))


def _poly_approx(s):
    #
    # p(s) = s + s**3/60 + s**5/1400 + s**7/25200 + 43*s**9/17248000 + ...
    #      = s*(1 + s**2*(1/60 + s**2*(1/1400 + s**2*(1/25200 + s**2*43/17248000)))
    #
    # See, for example, the wikipedia article "Kepler's equation"
    # (https://en.wikipedia.org/wiki/Kepler%27s_equation).  In particular, see the
    # series expansion for the inverse Kepler equation when the eccentricity e is 1.
    #
    # Here we include terms up to s**9.
    s2 = s**2
    return (s*(mpmath.mp.one
               + s2*(mpmath.mpf('1/60')
                     + s2*(mpmath.mpf('1/1400')
                           + s2*(mpmath.mpf('1/25200')
                                 + s2*mpmath.mpf('43/17248000'))))))


def invcdf(p):
    """
    Inverse of the CDF of the raised cosine distribution.
    """
    with mpmath.extradps(5):
        p = mpmath.mpf(p)

        if p < 0 or p > 1:
            return mpmath.nan
        if p == 0:
            return -mpmath.pi
        if p == 1:
            return mpmath.pi

        if p < 0.094:
            x = _poly_approx(mpmath.cbrt(12*mpmath.pi*p)) - mpmath.pi
        elif p > 0.906:
            x = mpmath.pi - _poly_approx(mpmath.cbrt(12*mpmath.pi*(1 - p)))
        else:
            y = mpmath.pi*(2*p - 1)
            y2 = y**2
            x = y * _p2(y2) / _q2(y2)

        solver = 'mnewton'
        x = mpmath.findroot(f=lambda t: cdf(t) - p,
                            x0=x,
                            df=lambda t: (1 + mpmath.cos(t))/(2*mpmath.pi),
                            df2 =lambda t: -mpmath.sin(t)/(2*mpmath.pi),
                            solver=solver)

        return x
