"""
Inverse chi-square distribution
-------------------------------

The probability density function for the inverse chi-square
distribution is

    f(x, nu) = 2**(-nu/2) / Gamma(nu/2) * x**(-nu/2 - 1) * exp(-1/(2*x))

See the Wikipedia article `"Inverse-chi-squared distribution"
<https://en.wikipedia.org/wiki/Inverse-chi-squared_distribution>`_
for more information.  The functions here implement the first
definition given in the wikipedia article.  That is, if X has the
chi-square distribution with nu degrees of freedom, then 1/X has the
inverse chi-square distribution with nu degrees of freedom.

"""

import re
import mpmath


# module docstring substitution
_math_expression = r"""
.. math::
          f(x, \\nu) = \\frac{2^{-\\nu/2}}{\\Gamma(\\nu/2)}
                       x^{-\\nu/2 - 1} e^{-1/(2x)}
"""
_docstring_re_subs = [
    (r'    f\(x,.*$', _math_expression, 0, re.MULTILINE),
    (' nu ', r' :math:`\\nu` ', 0, 0),
]


__all__ = ['pdf', 'logpdf', 'cdf', 'sf', 'mean', 'mode', 'variance']


def _validate_nu(nu):
    if nu <= 0:
        raise ValueError('nu must be positive')


def pdf(x, nu):
    """
    PDF for the inverse chi-square distribution.
    """
    _validate_nu(nu)
    if x <= 0:
        return mpmath.mp.zero
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        nu = mpmath.mpf(nu)
        hnu = nu/2
        p = (mpmath.power(2, -hnu) * x**(-hnu - 1) * mpmath.exp(-1/(2*x))
             / mpmath.gamma(hnu))
        return p


def logpdf(x, nu):
    """
    Logarithm of the PDF for the inverse chi-square distribution.
    """
    _validate_nu(nu)
    if x <= 0:
        return mpmath.ninf
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        nu = mpmath.mpf(nu)
        hnu = nu/2
        logp = (-hnu*mpmath.log(2) + (-hnu - 1)*mpmath.log(x) - 1/(2*x)
                - mpmath.loggamma(hnu))
        return logp


def cdf(x, nu):
    """
    CDF for the inverse chi-square distribution.
    """
    _validate_nu(nu)
    if x <= 0:
        return mpmath.mp.zero
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        nu = mpmath.mpf(nu)
        c = mpmath.gammainc(nu/2, a=1/(2*x), b=mpmath.inf, regularized=True)
    return c


def sf(x, nu):
    """
    Survival function for the inverse chi-square distribution.
    """
    _validate_nu(nu)
    if x <= 0:
        return mpmath.mp.one
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        nu = mpmath.mpf(nu)
        s = mpmath.gammainc(nu/2, a=0, b=1/(2*x), regularized=True)
    return s


def mean(nu):
    """
    Mean of the inverse chi-square distribution.

    For nu > 2, the mean is 1/(nu - 2).

    """
    _validate_nu(nu)
    with mpmath.extradps(5):
        nu = mpmath.mpf(nu)
        return mpmath.mp.one / (nu - 2) if nu > 2 else mpmath.nan


mean._docstring_re_subs = [
    (r'     *1.*2\)$',
     '\n'.join([r'.. math::',
                r'        \\frac{1}{\\nu - 2}',
                r'']),
     0, re.MULTILINE),
    (r'1/\(nu - 2\)', r':math:`1/(\\nu - 2)`', 0, 0),
    ('nu > 2', r':math:`\\nu > 2`', 0, 0),
]


def mode(nu):
    """
    Mode of the inverse chi-square distribution.

    The mode is max(k - 2, 0).
    """
    _validate_nu(nu)
    with mpmath.extradps(5):
        nu = mpmath.mpf(nu)
        return 1 / (nu + 2)


def variance(nu):
    """
    Variance of the inverse chi-square distribution.

    For nu > 4, the variance is

        2 / ((nu - 2)**2 (nu - 4))

    """
    _validate_nu(nu)
    with mpmath.extradps(5):
        nu = mpmath.mpf(nu)
        return 2/(nu - 2)**2 / (nu - 4) if nu > 4 else mpmath.nan


variance._docstring_re_subs = [
    (r'     *2.*4\)\)$',
     '\n'.join([r'.. math::',
                r'        \\frac{2}{(\\nu - 2)^2 (\\nu - 4)}',
                r'']),
     0, re.MULTILINE),
    ('nu > 4', r':math:`\\nu > 4`', 0, 0),
]
