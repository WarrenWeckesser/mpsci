"""
Raised cosine distribution
--------------------------

"""

import re
from mpmath import mp
from ._common import _validate_p


__all__ = ['pdf', 'logpdf', 'cdf', 'invcdf', 'sf', 'invsf',
           'mean', 'var', 'skewness', 'kurtosis']


# TO DO: Add location and scale parameters.


def pdf(x):
    """
    Probability density function (PDF) of the raised cosine distribution.

    The PDF of the raised cosine distribution is

        f(x) = (1 + cos(x))/(2*pi)

    on the interval (-pi, pi) and zero elsewhere.
    """
    with mp.extradps(5):
        x = mp.mpf(x)
        if x <= -mp.pi or x >= mp.pi:
            return mp.zero
        return (mp.cos(x) + 1)/(2*mp.pi)


_pdf_docstring_replace = r"""
    .. math::

       f(x) = \\frac{1 + \\cos(x)}{2\\pi}

"""

pdf._docstring_re_subs = [
    (r'    f.*\*pi\)', _pdf_docstring_replace, 0, re.DOTALL),
    (r'\(-pi, pi\)', r':math:`(-\\pi, \\pi)`', 0, 0),
]


def logpdf(x):
    """
    Natual logarithm of the PDF of the raised cosine distribution.

    The PDF of the raised cosine distribution is

        f(x) = (1 + cos(x))/(2*pi)

    on the interval (-pi, pi) and zero elsewhere.
    """
    with mp.extradps(5):
        if x <= -mp.pi or x >= mp.pi:
            return -mp.inf
        return mp.log1p(mp.cos(x)) - mp.log(2*mp.pi)


logpdf._docstring_re_subs = [
    (r'    f.*\*pi\)', _pdf_docstring_replace, 0, re.DOTALL),
    (r'\(-pi, pi\)', r':math:`(-\\pi, \\pi)`', 0, 0),
]


def cdf(x):
    """
    Cumulative distribution function (CDF) of the raised cosine distribution.

    The CDF of the raised cosine distribution is

        F(x) = (pi + x + sin(x))/(2*pi)
    """
    with mp.extradps(5):
        x = mp.mpf(x)
        if x <= -mp.pi:
            return mp.zero
        if x >= mp.pi:
            return mp.one
        return mp.mpf('1/2') + (x + mp.sin(x))/(2*mp.pi)


_cdf_docstring_replace = r"""
    .. math::

       F(x) = \\frac{\\pi + x + \\sin(x)}{2\\pi}

"""

cdf._docstring_re_subs = [
    (r'    F.*pi\)', _cdf_docstring_replace, 0, re.DOTALL),
]


# XXX coefficients in _p2 and _q2 are 64 bit floating point, not
# mpmath floats.

def _p2(t):
    t = mp.mpf(t)
    return (0.5
            + t*(-0.06532856457583547
                 + t*(0.0020893844847965047
                      + t*-1.0233693819385904e-05)))


def _q2(t):
    t = mp.mpf(t)
    return (1.0
            + t*(-0.15149046248500425
                 + t*(0.006293153604697265
                      + t*-6.042645518776793e-05)))


def _poly_approx(s):
    #
    # p(s) = s + s**3/60 + s**5/1400 + s**7/25200 + 43*s**9/17248000 + ...
    #      = s*(1
    #           + s**2*(1/60
    #                   + s**2*(1/1400
    #                           + s**2*(1/25200
    #                                   + s**2*43/17248000)))
    #
    # See, for example, the wikipedia article "Kepler's equation"
    # (https://en.wikipedia.org/wiki/Kepler%27s_equation).  In particular,
    # see the series expansion for the inverse Kepler equation when the
    # eccentricity e is 1.
    #
    # Here we include terms up to s**9.
    s2 = s**2
    return (s*(mp.one
               + s2*(mp.mpf('1/60')
                     + s2*(mp.mpf('1/1400')
                           + s2*(mp.mpf('1/25200')
                                 + s2*mp.mpf('43/17248000'))))))


def invcdf(p):
    """
    Inverse of the CDF of the raised cosine distribution.
    """
    with mp.extradps(5):
        p = _validate_p(p)
        if p == 0:
            return -mp.pi
        if p == 1:
            return mp.pi

        if p < 0.094:
            x = _poly_approx(mp.cbrt(12*mp.pi*p)) - mp.pi
        elif p > 0.906:
            x = mp.pi - _poly_approx(mp.cbrt(12*mp.pi*(1 - p)))
        else:
            y = mp.pi*(2*p - 1)
            y2 = y**2
            x = y * _p2(y2) / _q2(y2)

        solver = 'mnewton'
        x = mp.findroot(f=lambda t: cdf(t) - p,
                        x0=x,
                        df=lambda t: (1 + mp.cos(t))/(2*mp.pi),
                        df2=lambda t: -mp.sin(t)/(2*mp.pi),
                        solver=solver)

        return x


def sf(x):
    """
    Survival function for the cosine distribution.
    """
    return cdf(-x)


def invsf(p):
    """
    Inverse of the survival function of the cosine distribution.
    """
    return -invcdf(p)


def mean():
    """
    Mean of the cosine distribution.

    The mean is 0.
    """
    return mp.zero


def var():
    """
    Variance of the cosine distribution.

    The variance is pi**2/3 - 2.
    """
    return mp.pi**2/3 - 2


def skewness():
    """
    Skewness of the cosine distribution.

    The skewness is 0.
    """
    return mp.zero


def kurtosis():
    """
    Excess kurtosis of the cosine distribution.

    The excess kurtosis is (6/5)*(90 - pi**4)/(pi**2 - 6)**2.
    """
    pi2 = mp.pi**2
    return (90 - pi2**2)/(pi2 - 6)**2 * 6 / 5
