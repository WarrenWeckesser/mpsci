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

    If any value in x is 0, the return value is 0.

    hmean accepts negative values. Usually the harmonic mean is defined
    for positive values only, but the formula is well-defined as long as
    1/x[0] + 1/x[1] + ... + 1/x[-1] is not 0.

    If that expression is 0, and the signs of the x values are mixed, nan
    is returned.  If the signs are not mixed, then either all the values are
    +inf or they are all -inf.  For those cases, +inf and -inf are returned,
    respectively.

    Examples
    --------
    >>> from mpsci.stats import hmean
    >>> import mpmath
    >>> mpmath.mp.dps = 25

    >>> hmean([1, 3, 3])
    mpf('1.8')

    >>> hmean([10, 3, -2])
    mpf('-45.0')

    >>> hmean(range(1, 10))
    mpf('3.181371861411137606957497545')

    >>> hmean([2, -2])
    mpf('nan')

    >>> hmean([mpmath.inf, mpmath.inf, mpmath.inf])
    mpf('+inf')

    >>> hmean([mpmath.inf, mpmath.inf, -mpmath.inf])
    mpf('nan')

    >>> hmean([-mpmath.inf, -mpmath.inf, -mpmath.inf])
    >>> mpf('-inf')

    """
    npos = 0
    nneg = 0
    nzero = 0
    for t in x:
        if t > 0:
            npos += 1
        elif t < 0:
            nneg += 1
        else:
            nzero += 1
    if nzero > 0:
        return mpmath.mp.zero
    mixed_signs = npos > 0 and nneg > 0
    with mpmath.extraprec(16):
        m = mean([1/mpmath.mpf(t) for t in x])
        if m == 0:
            if mixed_signs:
                return mpmath.mp.nan
            elif npos > 0:
                return mpmath.mp.inf
            else:
                return -mpmath.mp.inf
        else:
            return 1 / m



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
