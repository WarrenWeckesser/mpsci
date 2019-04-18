"""
Basic statistics
----------------
mean
    Arithmetic mean, with optional weights
var
    Variance
std
    Standard deviation
gmean
    Geometric mean
hmean
    Harmonic mean
pmean
    Power (or generalized) mean
"""

import mpmath


__all__ = ['mean', 'var', 'std', 'gmean', 'hmean', 'pmean']


# XXX Add special handling for inf and nan in these functions?

def mean(x, weights=None):
    """
    Mean of the values in the sequence x.

    Negative values in weights are allowed.  The only constraint is
    that sum(weights) is not zero.
    """
    n = len(x)
    if weights is not None:
        if len(weights) != n:
            raise ValueError('x and weights must have the same length.')

    with mpmath.extraprec(16):
        if weights is None:
            return mpmath.fsum(x) / len(x)
        else:
            if mpmath.fsum(weights) == 0:
                raise ZeroDivisionError('sum(weights) must be nonzero.')
            return mpmath.fsum(t*w for t, w in zip(x, weights)) / mpmath.fsum(weights)


def var(x, ddof=0):
    """
    Variance of the values in the sequence x.
    """
    n = len(x)
    with mpmath.extraprec(16):
        sumx = mpmath.fsum(x)
        meanx = sumx / n
        varx = mpmath.fsum((mpmath.mpf(t) - meanx)**2 for t in x)/(n - ddof)
    return varx


def std(x, ddof=0):
    """
    Standard deviation of the values in the sequence x.
    """
    with mpmath.extraprec(16):
        return mpmath.sqrt(var(x, ddof))


def gmean(x):
    """
    Geometric mean of the values in the sequence x.

    All the values in x must be nonnegative.
    """
    if any(t < 0 for t in x):
        raise ValueError("all values in x must be nonnegative.")
    if 0 in x:
        return mpmath.mp.zero
    with mpmath.extraprec(16):
        return mpmath.exp(mean([mpmath.log(t) for t in x]))


def hmean(x):
    """
    Harmonic mean of the values in the sequence x.

    All the values in x must be greater than zero.
    """
    if any(t <= 0 for t in x):
        raise ValueError("all values in x must be greater than zero.")
    with mpmath.extraprec(16):
        return 1 / mean([1/mpmath.mpf(t) for t in x])



def pmean(x, p):
    """
    Power (or generalized) mean of the values in the sequence x.
    """
    # Special cases
    if p == 0:
        return gmean(x)
    elif p == 1:
        return mean(x)
    elif p == -1:
        return hmean(x)
    elif mpmath.isinf(p):
        with mpmath.extraprec(16):
            if p > 0:
                return max(mpmath.mpf(t) for t in x)
            else:
                return min(mpmath.mpf(t) for t in x)

    with mpmath.extraprec(16):
        p = mpmath.mpf(p)
        return mpmath.power(mean([mpmath.mpf(t)**p for t in x]), 1/p)
