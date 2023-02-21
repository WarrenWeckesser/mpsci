"""
Generalized inverse Gaussian distribution
-----------------------------------------
"""

import re
from mpmath import mp


def _validate_params(p, b, loc, scale):
    if b <= 0:
        raise ValueError('b must be greater than 0')
    if scale <= 0:
        raise ValueError('scale must be greater than 0')
    return mp.mpf(p), mp.mpf(b), mp.mpf(loc), mp.mpf(scale)


__all__ = ['pdf', 'logpdf', 'cdf', 'sf', 'mean', 'mode', 'entropy']


_latex_pdf = r"""
.. math::

   \\frac{z^{(p - 1)}\\exp\\left(-\\frac{b}{2}\\left(z + \\frac{1}{z}\\right)\\right)}
         {s K_{p}(b)}

"""

_pdf_docstring_re_subs = [
    (r'[ ]+z.*\* K_p\(b\)', _latex_pdf, 0, re.DOTALL),
    (r'where s', r'where :math:`s`', 0, 0),
    (r'z = \(x - loc\)/s', r':math:`z = (x - \\textsf{loc})/s`', 0, 0),
    (r'K_p\(b\) is', r':math:`K_p(b)` is', 0, 0),
    (r'x > loc', r':math:`x > \\textsf{loc}`', 0, 0),
    (r'x <= loc', r':math:`x \\le \\textsf{loc}`', 0, 0),
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
    with mp.extradps(5):
        p, b, loc, scale = _validate_params(p, b, loc, scale)
        x = mp.mpf(x)
        if x <= loc:
            return mp.zero
        z = (x - loc)/scale
        return (mp.power(z, p - 1)
                * mp.exp(-b*(z + 1/z)/2)
                / (2*mp.besselk(p, b))
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
    with mp.extradps(5):
        p, b, loc, scale = _validate_params(p, b, loc, scale)
        x = mp.mpf(x)

        if x <= loc:
            return mp.ninf
        z = (x - loc)/scale
        return ((p - 1)*mp.log(z)
                - b*(z + 1/z)/2
                - mp.log(2*mp.besselk(p, b))
                - mp.log(scale))


logpdf._docstring_re_subs = _pdf_docstring_re_subs


def cdf(x, p, b, loc=0, scale=1):
    """
    Cumulative distribution function of the generalized inverse Gaussian
    distribution.

    The CDF is computed by using mpmath.quad to numerically integrate the PDF.
    """
    with mp.extradps(5):
        p, b, loc, scale = _validate_params(p, b, loc, scale)
        x = mp.mpf(x)

        if x <= loc:
            return mp.zero
        m = mode(p, b, loc, scale)
        # If the mode is in the integration interval, use it to do the integral
        # in two parts.  Otherwise do just one integral.
        if x <= m:
            c = mp.quad(lambda t: pdf(t, p, b, loc, scale), [loc, x])
        else:
            c = (mp.quad(lambda t: pdf(t, p, b, loc, scale), [loc, m]) +
                 mp.quad(lambda t: pdf(t, p, b, loc, scale), [m, x]))
        c = min(c, mp.one)
        return c


def sf(x, p, b, loc=0, scale=1):
    """
    Survival function of the generalized inverse Gaussian distribution.

    The survival function is computed by using mpmath.quad to numerically
    integrate the PDF.
    """
    with mp.extradps(5):
        p, b, loc, scale = _validate_params(p, b, loc, scale)
        x = mp.mpf(x)

        if x <= loc:
            return mp.one
        m = mode(p, b, loc, scale)
        # If the mode is in the integration interval, use it to do the integral
        # in two parts.  Otherwise do just one integral.
        if x >= m:
            s = mp.quad(lambda t: pdf(t, p, b, loc, scale), [x, mp.inf])
        else:
            s = (mp.quad(lambda t: pdf(t, p, b, loc, scale), [x, m]) +
                 mp.quad(lambda t: pdf(t, p, b, loc, scale), [m, mp.inf]))
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
    with mp.extradps(5):
        p, b, loc, scale = _validate_params(p, b, loc, scale)
        return loc + scale*mp.besselk(p + 1, b)/mp.besselk(p, b)


_mean_latex = r"""
.. math::

   \\textsf{loc} + \\textsf{scale} \\frac{K_{p+1}(b)}{K_p(b)}

"""

mean._docstring_re_subs = [
    (r'[ ]+K_.*K_p\(b\)', _mean_latex, 0, re.DOTALL),
    (r'where K_n\(x\)', r'where :math:`K_n(x)`', 0, 0),
]


def mode(p, b, loc=0, scale=1):
    """
    Mode of the generalized inverse Gaussian distribution.

    The mode is:

                    p - 1 + sqrt((p - 1)**2 + b**2)
        loc + scale -------------------------------
                                  b
    """
    with mp.extradps(5):
        p, b, loc, scale = _validate_params(p, b, loc, scale)
        return loc + scale*(p - 1 + mp.sqrt((p - 1)**2 + b**2))/b


_mode_latex = r"""
.. math::

   \\textsf{loc} +
   \\textsf{scale} \\frac{p - 1 + \\sqrt{(p - 1)^2 + b^2}}{b}

"""

mode._docstring_re_subs = [
    (r'[ ]+p.*  b', _mode_latex, 0, re.DOTALL),
]


def _besselk_nderiv(n, k):
    return mp.diff(lambda n: mp.besselk(n, k), n)


def entropy(p, b, loc=0, scale=1):
    """
    Differential entropy of the generalized inverse Gaussian distribution.
    """
    with mp.extradps(5):
        p, b, loc, scale = _validate_params(p, b, loc, scale)

        # See, for example,
        # https://en.wikipedia.org/wiki/Generalized_inverse_Gaussian_distribution
        # for the entropy formula.
        kpb = mp.besselk(p, b)
        t1 = mp.log(scale)
        t2 = mp.log(2*kpb)
        t3 = -(p - 1)*_besselk_nderiv(p, b)/kpb
        t4 = b/(2*kpb)*(mp.besselk(p + 1, b) + mp.besselk(p - 1, b))
        return mp.fsum([t1, t2, t3, t4])
