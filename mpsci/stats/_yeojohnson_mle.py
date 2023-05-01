
import re
from mpmath import mp
from ._yeojohnson import yeojohnson
from ._basic import var


def yeojohnson_llf(lam, x):
    r"""The log-likelihood function for the Yeo-Johnson transformation.

    Parameters
    ----------
    lam : scalar
        Parameter for Yeo-Johnson transformation. See `yeojohnson` for
        details.
    x : sequence of numbers
        Data to calculate Yeo-Johnson log-likelihood for.

    Returns
    -------
    llf : float
        Yeo-Johnson log-likelihood of `x` given `lam`.

    Notes
    -----
    The Yeo-Johnson log-likelihood is

        llf = -N/2 * log(sigma**2) + (lam - 1)*sum(sign(x)*log(abs(x) + 1)

    where sigma**2 is the estimated variance of the Yeo-Johnson
    transformed input data ``x``.

    """
    with mp.extradps(5):
        lam = mp.mpf(lam)
        n = len(x)
        x = [mp.mpf(t) for t in x]
        if n == 0:
            raise ValueError('x must have at least one element')

        y = [yeojohnson(t, lam) for t in x]
        y_var = var(y)
        log1psum = mp.fsum([mp.sign(t)*mp.log1p(abs(t)) for t in x])
        llf = -n/2 * mp.log(y_var) + (lam - 1)*log1psum
        return llf


_llf_expression = r"""
.. math::

    \\frac{-N}{2} \\log\\left(\\hat{\\sigma}^2\\right)
    + (\\lambda - 1)\\sum_{i}\\textrm{sign}(x_i) \\log(|x_i| + 1)

"""

yeojohnson_llf._docstring_re_subs = [
    (r'llf =.*?\+ 1\)', _llf_expression, 0, re.MULTILINE),
    (r'sigma\*\*2', r':math:`N` is ``len(x)`` and :math:`\\hat{\\sigma}^2`',
     0, 0),
]


def yeojohnson_mle(x, lam0=1):
    """Compute the maximum likelihood estimate of the Yeo-Johnson parameter.

    `x` must be a sequence of numbers.

    `lam0` is the initial guess for the optimal parameter.  The default
    value, `lam0=1`, is generally not a good initial guess, and the may
    have to be changed for the method to find the optimal parameters. The
    algorithm uses numerical differentiation and numerical root-finding,
    and it might fail to converge in some cases.  If that happens, try
    changing `lam0`.  Even changing only the sign of `lam0` can make a
    difference.
    """
    lam = mp.findroot(lambda t: mp.diff(lambda s: yeojohnson_llf(s, x), t),
                      lam0)
    return lam
