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
from mpmath import mp


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


__all__ = ['pdf', 'logpdf', 'cdf', 'sf', 'mean', 'mode', 'var']


def _validate_nu(nu):
    if nu <= 0:
        raise ValueError('nu must be positive')
    return mp.mpf(nu)


def pdf(x, nu):
    """
    PDF for the inverse chi-square distribution.
    """
    with mp.extradps(5):
        nu = _validate_nu(nu)
        x = mp.mpf(x)
        if x <= 0:
            return mp.zero
        hnu = nu/2
        p = (mp.power(2, -hnu) * x**(-hnu - 1) * mp.exp(-1/(2*x))
             / mp.gamma(hnu))
        return p


def logpdf(x, nu):
    """
    Logarithm of the PDF for the inverse chi-square distribution.
    """
    with mp.extradps(5):
        nu = _validate_nu(nu)
        x = mp.mpf(x)
        if x <= 0:
            return mp.ninf
        hnu = nu/2
        logp = (-hnu*mp.log(2) + (-hnu - 1)*mp.log(x) - 1/(2*x)
                - mp.loggamma(hnu))
        return logp


def cdf(x, nu):
    """
    CDF for the inverse chi-square distribution.
    """
    with mp.extradps(5):
        nu = _validate_nu(nu)
        x = mp.mpf(x)
        if x <= 0:
            return mp.zero
        c = mp.gammainc(nu/2, a=1/(2*x), b=mp.inf, regularized=True)
    return c


def sf(x, nu):
    """
    Survival function for the inverse chi-square distribution.
    """
    with mp.extradps(5):
        nu = _validate_nu(nu)
        x = mp.mpf(x)
        if x <= 0:
            return mp.one
        s = mp.gammainc(nu/2, a=0, b=1/(2*x), regularized=True)
    return s


def mean(nu):
    """
    Mean of the inverse chi-square distribution.

    For nu > 2, the mean is 1/(nu - 2).

    """
    with mp.extradps(5):
        nu = _validate_nu(nu)
        return mp.one / (nu - 2) if nu > 2 else mp.nan


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
    with mp.extradps(5):
        nu = _validate_nu(nu)
        return 1 / (nu + 2)


def var(nu):
    """
    Variance of the inverse chi-square distribution.

    For nu > 4, the variance is

        2 / ((nu - 2)**2 (nu - 4))

    """
    with mp.extradps(5):
        nu = _validate_nu(nu)
        return 2/(nu - 2)**2 / (nu - 4) if nu > 4 else mp.nan


var._docstring_re_subs = [
    (r'     *2.*4\)\)$',
     '\n'.join([r'.. math::',
                r'        \\frac{2}{(\\nu - 2)^2 (\\nu - 4)}',
                r'']),
     0, re.MULTILINE),
    ('nu > 4', r':math:`\\nu > 4`', 0, 0),
]
