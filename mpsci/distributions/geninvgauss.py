"""
Generalized inverse Gaussian distribution
-----------------------------------------
"""

import re
import mpmath


__all__ = ['pdf', 'logpdf', 'cdf', 'sf', 'mean', 'mode']


_latex_pdf = r"""
.. math::

   \\frac{z^{(p - 1)}\\exp\\left(-\\frac{b}{2}\\left(z + \\frac{1}{z}\\right)\\right)}
         {s K_{p}(b)}

"""

_pdf_docstring_re_subs = [
    ('[ ]+z.*\* K_p\(b\)', _latex_pdf, 0, re.DOTALL),
    ('where s', r'where :math:`s`', 0, 0),
    ('z = \(x - loc\)/s', r':math:`z = (x - \\textsf{loc})/s`', 0, 0),
    ('K_p\(b\) is', r':math:`K_p(b)` is', 0, 0),
    ('x > loc', r':math:`x > \\textsf{loc}`', 0, 0),
    ('x <= loc', r':math:`x \\le \\textsf{loc}`', 0, 0),
]


# Parameters have been chosen to match the proposed implementation of
# geninvgauss in scipy.

def pdf(x, p, b, loc=0, scale=1):
    """
    Probability density function of the generalized inverse Gaussian
    distribution.

    The PDF for x > loc is:

        z**(p - 1) * exp(-b*(z + 1/z)/2))
        ---------------------------------
                 s * K_p(b)

    where s is the scale, z = (x - loc)/s, and K_p(b) is the modified Bessel
    function of the second kind.  For x <= loc, the PDF is zero.
    """
    x = mpmath.mpf(x)
    p = mpmath.mpf(p)
    b = mpmath.mpf(b)
    loc = mpmath.mpf(loc)
    scale = mpmath.mpf(scale)

    if x <= loc:
        return mpmath.mp.zero
    z = (x - loc)/scale
    return (mpmath.power(z, p - 1)
            * mpmath.exp(-b*(z + 1/z)/2)
            / (2*mpmath.besselk(p, b))
            / scale)


pdf._docstring_re_subs = _pdf_docstring_re_subs


def logpdf(x, p, b, loc=0, scale=1):
    """
    Log of the PDF of the generalized inverse Gaussian distribution.

    The PDF for x > loc is:

        z**(p - 1) * exp(-b*(z + 1/z)/2))
        ---------------------------------
                 scale * K_p(b)

    where s is the scale, z = (x - loc)/s, and K_p(b) is the modified Bessel
    function of the second kind.  For x <= loc, the PDF is zero.
    """
    x = mpmath.mpf(x)
    p = mpmath.mpf(p)
    b = mpmath.mpf(b)
    loc = mpmath.mpf(loc)
    scale = mpmath.mpf(scale)

    if x <= loc:
        return -mpmath.mp.inf
    z = (x - loc)/scale
    return ((p - 1)*mpmath.log(z)
            - b*(z + 1/z)/2
            - mpmath.log(2*mpmath.besselk(p, b))
            - mpmath.log(scale))


logpdf._docstring_re_subs = _pdf_docstring_re_subs


def cdf(x, p, b, loc=0, scale=1):
    """
    Cumulative distribution function of the generalized inverse Gaussian
    distribution.

    The CDF is computed by using mpmath.quad to numerically integrate the PDF.
    """
    x = mpmath.mpf(x)
    p = mpmath.mpf(p)
    b = mpmath.mpf(b)
    loc = mpmath.mpf(loc)
    scale = mpmath.mpf(scale)

    if x <= loc:
        return mpmath.mp.zero
    m = mode(p, b, loc, scale)
    # If the mode is in the integration interval, use it to do the integral
    # in two parts.  Otherwise do just one integral.
    if x <= m:
        c = mpmath.quad(lambda t: pdf(t, p, b, loc, scale), [loc, x])
    else:
        c = (mpmath.quad(lambda t: pdf(t, p, b, loc, scale), [loc, m]) +
             mpmath.quad(lambda t: pdf(t, p, b, loc, scale), [m, x]))
    c = min(c, mpmath.mp.one)
    return c


def sf(x, p, b, loc=0, scale=1):
    """
    Survival function of the generalized inverse Gaussian distribution.

    The survival function is computed by using mpmath.quad to numerically
    integrate the PDF.
    """
    x = mpmath.mpf(x)
    p = mpmath.mpf(p)
    b = mpmath.mpf(b)
    loc = mpmath.mpf(loc)
    scale = mpmath.mpf(scale)

    if x <= loc:
        return mpmath.mp.one
    m = mode(p, b, loc, scale)
    # If the mode is in the integration interval, use it to do the integral
    # in two parts.  Otherwise do just one integral.
    if x >= m:
        s = mpmath.quad(lambda t: pdf(t, p, b, loc, scale), [x, mpmath.inf])
    else:
        s = (mpmath.quad(lambda t: pdf(t, p, b, loc, scale), [x, m]) +
             mpmath.quad(lambda t: pdf(t, p, b, loc, scale), [m, mpmath.inf]))
    return s


def mean(p, b, loc=0, scale=1):
    """
    Mean of the generalized inverse Gaussian distribution.

    The mean is:

                     K_{p + 1}(b)
        loc + scale --------------
                        K_p(b)

    where K_n(x) is the modified Bessel function of the second kind
    (implemented in mpmath as besselk(n, x)).
    """
    p = mpmath.mpf(p)
    b = mpmath.mpf(b)
    loc = mpmath.mpf(loc)
    scale = mpmath.mpf(scale)

    return loc + scale*mpmath.besselk(p + 1, b)/mpmath.besselk(p, b)


_mean_latex = r"""
.. math::

   \\textsf{loc} + \\textsf{scale} \\frac{K_{p+1}(b)}{K_p(b)}

"""

mean._docstring_re_subs = [
    ('[ ]+K_.*K_p\(b\)', _mean_latex, 0, re.DOTALL),
    ('where K_n\(x\)', r'where :math:`K_n(x)`', 0, 0),
]

def mode(p, b, loc=0, scale=1):
    """
    Mode of the generalized inverse Gaussian distribution.

    The mode is:

                    p - 1 + sqrt((p - 1)**2 + b**2)
        loc + scale -------------------------------
                                  b
    """
    p = mpmath.mpf(p)
    b = mpmath.mpf(b)
    loc = mpmath.mpf(loc)
    scale = mpmath.mpf(scale)

    return loc + scale*(p - 1 + mpmath.sqrt((p - 1)**2 + b**2))/b

_mode_latex = r"""
.. math::

   \\textsf{loc} +
   \\textsf{scale} \\frac{p - 1 + \\sqrt{(p - 1)^2 + b^2}}{b}

"""

mode._docstring_re_subs = [
    ('[ ]+p.*  b', _mode_latex, 0, re.DOTALL),
]
