"""
Code for the maximum likelihood estimation of the
Box-Cox transformation.
"""

from mpmath import mp
from ._basic import var


def boxcox_llf(lam, x):
    """
    Log-likelihood for maximum likelihood estimation of the Box-Cox parameters.
    """
    with mp.extradps(5):
        lam = mp.mpf(lam)
        x = [mp.mpf(t) for t in x]
        n = len(x)

        logdata = [mp.log(t) for t in x]
        sumlogdata = mp.fsum(logdata)

        # Compute the variance of the transformed data.
        if lam == 0:
            variance = var(logdata)
        else:
            # Transform without the constant offset 1/lam.  The offset does
            # not effect the variance, and the subtraction of the offset can
            # lead to loss of precision.
            variance = var([t**lam / lam for t in x])

        return (lam - 1) * sumlogdata - n/2 * mp.log(variance)


def _boxcox_llf_deriv(lam, x):
    """
    Use mpmath.diff to estimate the derivative of boxcox_llf w.r.t. lam.
    """
    with mp.extradps(10):
        return mp.diff(lambda t: boxcox_llf(t, x), lam)


def boxcox_mle(x, lam0=0):
    """
    Maximum likelihood estimate of lambda in the Box-Cox transformation.

    Parameters
    ----------
    x : sequence of numbers
        The dataset for which the MLE of lambda is to be estimated.
    lam0 : float
        The initial guess for the numerical procedure that computes lambda.

    Returns
    -------
    lam : mpmath.mpf
        The estimate of the Box-Cox lambda parameter.

    Examples
    --------
    >>> from mpmath import mp
    >>> mp.dps = 40

    >>> from mpsci.stats import boxcox_mle
    >>> x = [12.5, 13, 18.2, 20, 24.9, 25.3, 32.8]
    >>> boxcox_mle(x)
    mpf('0.1902293418567596520673395974496886925599866')

    """
    return mp.findroot(lambda t: _boxcox_llf_deriv(t, x), lam0)
