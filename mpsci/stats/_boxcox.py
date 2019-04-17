"""
Some functions for testing the maximum likelihood estimate of the
Box-Cox transformation.

This code is not in the public API yet.  It has not been thoroughly
tested, and there are some improvements that can be made.
"""
import mpmath


def _var(x):
    n = len(x)
    sumx = mpmath.fsum(x)
    meanx = sumx / n
    varx = mpmath.fsum((mpmath.mpf(t) - meanx)**2 for t in x)/n
    return varx


def boxcox_llf(lmb, x):
    lmb = mpmath.mpf(lmb)
    x = [mpmath.mpf(t) for t in x]
    n = len(x)

    logdata = [mpmath.log(t) for t in x]
    sumlogdata = mpmath.fsum(logdata)

    # Compute the variance of the transformed data.
    if lmb == 0:
        variance = _var(logdata)
    else:
        # Transform without the constant offset 1/lmb.  The offset does
        # not effect the variance, and the subtraction of the offset can
        # lead to loss of precision.
        variance = _var([t**lmb / lmb for t in x])

    return (lmb - 1) * sumlogdata - n/2 * mpmath.log(variance)



def _boxcox_llf_deriv_not_finished(lmb, x):
    """
    This function assumes lmb != 0.
    """
    lmb = mpmath.mpf(lmb)
    x = [mpmath.mpf(t) for t in x]
    n = len(x)

    logdata = [mpmath.log(t) for t in x]
    sumlogdata = mpmath.fsum(logdata)

    mean_deriv_bc_sq = mpmath.fsum(2*(t**lmb/lmb)**2*(mpmath.log(t) - 1/lmb) for t in x) / n
    mean_bc_data = mpmath.fsum(t**lmb/lmb for t in x) / n
    deriv_mean_bc_data = mpmath.fsum(t**lmb/lmb * (mpmath.log(t) - 1/lmb) for t in x) / n

    dvar = mean_deriv_bc_sq - 2 * mean_bc_data * deriv_mean_bc_data

    return sumlogdata - n/2 * dvar



def _boxcox_llf_deriv(lmb, x):
    """
    Use mpmath.diff to estimate the derivative of _boxcox_llf w.r.t. lmb.
    """
    with mpmath.extradps(10):
        return mpmath.diff(lambda t: _boxcox_llf(t, x), lmb)


def boxcox_mle(x):
    """
    Maximum likelihood estimate of lambda in the Box-Cox transformation.
    """
    return mpmath.findroot(lambda t: _boxcox_llf_deriv(t, x), 0.0)
