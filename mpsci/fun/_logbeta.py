
import re
import mpmath


__all__ = ['logbeta']


def logbeta(x, y):
    """
    Natural logarithm of beta(x, y).

    The beta function is

                     Gamma(x) Gamma(y)
        beta(x, y) = -----------------
                       Gamma(x + y)

    where Gamma(z) is the Gamma function.
    """
    with mpmath.extradps(5):
        mx = mpmath.mpf(x)
        my = mpmath.mpf(y)
        return (mpmath.loggamma(x)
                + mpmath.loggamma(y)
                - mpmath.loggamma(mx + my))


_beta_func_latex = r"""
    .. math::

        B(x, y) = \\frac{\\Gamma(x)\\Gamma(y)}{\\Gamma(x + y)}
"""

logbeta._docstring_re_subs = [
    (r'     *Gamma\(x.*x \+ y\)', _beta_func_latex, 0, re.DOTALL),
    (r' Gamma\(z\)', r' :math:`\\Gamma(z)`', 0, 0),
    (r' beta\(x, y\)', r' :math:`B(x, y)`', 0, 0),
]
