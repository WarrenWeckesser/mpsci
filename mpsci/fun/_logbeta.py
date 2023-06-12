
import re
from mpmath import mp


__all__ = ['logbeta', 'multivariate_logbeta']


def logbeta(x, y):
    """
    Natural logarithm of beta(x, y).

    The beta function is

                     Gamma(x) Gamma(y)
        beta(x, y) = -----------------
                       Gamma(x + y)

    where Gamma(z) is the Gamma function.

    Examples
    --------
    >>> from mpmath import mp
    >>> from mpsci.distributions import logbeta

    >>> mp.dps = 25

    >>> logbeta(mp.pi, 1.25)
    mpf('-1.575224779107371741939220563')

    >>> mp.log(mp.beta(mp.pi, 1.25))
    mpf('-1.575224779107371741939220592')

    """
    with mp.extradps(5):
        return (mp.loggamma(x) + mp.loggamma(y) - mp.loggamma(mp.fsum([x, y])))


_beta_func_latex = r"""
    .. math::

        B(x, y) = \\frac{\\Gamma(x)\\Gamma(y)}{\\Gamma(x + y)}
"""

logbeta._docstring_re_subs = [
    (r'     *Gamma\(x.*x \+ y\)', _beta_func_latex, 0, re.DOTALL),
    (r' Gamma\(z\)', r' :math:`\\Gamma(z)`', 0, 0),
    (r' beta\(x, y\)', r' :math:`B(x, y)`', 0, 0),
]


def multivariate_logbeta(alpha):
    """
    Logarithm of the multivariate beta function.

    The multivariate beta function for the vector alpha is

                   Sum[Gamma(alpha_k)]
        B(alpha) = -----------------
                   Gamma(Sum(alpha))

    where Gamma(z) is the Gamma function.

    Examples
    --------
    >>> from mpmath import mp
    >>> mp.dps = 25
    >>> from mpsci.fun import multivariate_logbeta

    >>> multivariate_logbeta([1.5, 2.0, 4.0, 1.25, 5.0])
    mpf('-17.15300326630985105656599532')
    """
    with mp.extradps(5):
        terms = [mp.loggamma(t) for t in alpha]
        terms.append(-mp.loggamma(mp.fsum(alpha)))
        return mp.fsum(terms)


_mvbeta_func_latex = r"""
    .. math::

        B(\\alpha) = \\frac{\\sum_{k}\\Gamma(\\alpha_{k})}{\\Gamma(\\sum_{k}\\alpha_{k})}
"""

multivariate_logbeta._docstring_re_subs = [
    (r' *Sum.*alpha\)\)', _mvbeta_func_latex, 0, re.DOTALL),
    (r' Gamma\(z\)', r' :math:`\\Gamma(z)`', 0, 0),
    (r'vector alpha', r'vector :math:`\\alpha`', 0, 0),
]
