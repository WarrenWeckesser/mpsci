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
lehmer_mean
    Lehmer mean

Other utility functions
-----------------------
unique_counts
    Return the unique values in a sequence and the number of
    occurrences of each unique value.
"""

from itertools import groupby
from mpmath import mp
from ..fun import xlogy as _xlogy


__all__ = ['mean', 'var', 'std', 'variation',
           'gmean', 'hmean', 'pmean', 'lehmer_mean',
           'unique_counts']


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

    with mp.extraprec(16):
        if weights is None:
            return mp.fsum(x) / len(x)
        else:
            if mp.fsum(weights) == 0:
                raise ZeroDivisionError('sum(weights) must be nonzero.')
            return (mp.fsum(t*w for t, w in zip(x, weights)) /
                    mp.fsum(weights))


def var(x, ddof=0):
    """
    Variance of the values in the sequence x.
    """
    n = len(x)
    with mp.extraprec(16):
        sumx = mp.fsum(x)
        meanx = sumx / n
        varx = mp.fsum((mp.mpf(t) - meanx)**2 for t in x)/(n - ddof)
    return varx


def std(x, ddof=0):
    """
    Standard deviation of the values in the sequence x.
    """
    with mp.extraprec(16):
        return mp.sqrt(var(x, ddof))


def variation(x, ddof=1):
    """
    The variation of x.

    The variation the ratio of the standard deviation to the mean:

        std(x, ddof) / mean(x)

    Note that, unlike ``var`` and ``std``, the default value of ``ddof`` is 1.
    This is the more typical value used when computing the variation.

    (The implementation is simply std(x, ddof) / mean(x); no special
    handling is provided for `nan` values, a mean of 0, etc.)

    Examples
    --------
    >>> from mpsci.stats import variation
    >>> variation([2, 3, 5, 8, 13, 21])
    >>> mpf('0.83418102841390518')

    For comparison to `scipy.stats.variation`, use `ddof=0`:

    >>> variation([2, 3, 5, 8, 13, 21], ddof=0)
    >>> mpf('0.76149961050858964')
    """
    with mp.extraprec(16):
        s = std(x, ddof=ddof)
        m = mean(x)
        return s / m


def gmean(x, *, weights=None):
    """
    Geometric mean of the values in the sequence x.

    All the values in x must be nonnegative.

    If weights is not None, it must be a sequence with the same length
    as x.  The sum of weights must not be zero.
    """
    if any(t < 0 for t in x):
        raise ValueError("all values in x must be nonnegative.")
    with mp.extraprec(16):
        if weights is None:
            if 0 in x:
                return mp.zero
            return mp.exp(mean([mp.log(t) for t in x]))
        else:
            # Weighted geometric mean
            wsum = mp.fsum(weights)
            if wsum == 0:
                raise ValueError('sum of weights must not be 0.')
            wlogxsum = mp.fsum([_xlogy(wi, xi)
                                for (xi, wi) in zip(x, weights)])
            return mp.exp(wlogxsum / wsum)


def hmean(x, *, weights=None):
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
    >>> from mpmath import mp
    >>> mp.dps = 25

    >>> hmean([1, 3, 3])
    mpf('1.8')

    >>> hmean([1, 3, 3], weights=[5, 1, 2])
    mpf('1.333333333333333333333333333')

    >>> hmean([10, 3, -2])
    mpf('-45.0')

    >>> hmean(range(1, 10))
    mpf('3.181371861411137606957497545')

    >>> hmean([2, -2])
    mpf('nan')

    >>> hmean([mp.inf, mp.inf, mp.inf])
    mpf('+inf')

    >>> hmean([mp.inf, mp.inf, -mp.inf])
    mpf('nan')

    >>> hmean([-mp.inf, -mp.inf, -mp.inf])
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
        return mp.zero
    mixed_signs = npos > 0 and nneg > 0
    with mp.extraprec(16):
        m = mean([1/mp.mpf(t) for t in x], weights=weights)
        if m == 0:
            if mixed_signs:
                return mp.nan
            elif npos > 0:
                return mp.inf
            else:
                return -mp.inf
        else:
            return 1 / m


def pmean(x, *, p, weights=None):
    """
    Power (or generalized) mean of the values in the sequence x.
    """
    # Special cases
    if p == 0:
        return gmean(x, weights=weights)
    elif p == 1:
        return mean(x, weights=weights)
    elif p == -1:
        return hmean(x, weights=weights)
    elif mp.isinf(p):
        with mp.extraprec(16):
            if p > 0:
                return max(mp.mpf(t) for t in x)
            else:
                return min(mp.mpf(t) for t in x)

    with mp.extraprec(16):
        p = mp.mpf(p)
        return mp.power(mean([mp.mpf(t)**p for t in x], weights=weights), 1/p)


def lehmer_mean(x, *, p, weights=None):
    if any(t <= 0 for t in x):
        raise ValueError('All values in x must be positive.')
    if p == 0:
        return hmean(x, weights=weights)
    if p == 0.5:
        return gmean(x, weights=weights)
    if p == 1:
        return mean(x, weights=weights)
    p = mp.mpf(p)
    if mp.isinf(p):
        x = [mp.mpf(t) for t in x]
        if p < 0:
            return min(x)
        else:
            return max(x)
    # x = [mp.mpf(t) for t in x]
    return (mean([t**p for t in x], weights=weights) /
            mean([t**(p - 1) for t in x], weights=weights))


def unique_counts(x):
    """
    Unique values and their counts in the sequence `x`.

    Returns two tuples, The first is the sorted sequence of unique
    values in `x`, and the second is a sequence of integers (same
    length as the first) that gives the number of occurrences of the
    corresponding value.

    Examples
    --------
    >>> from mpsci.stats import unique_counts
    >>> x = [3, 4, 7, 8, 0, 0, 0, 7, 3, 2, 0, 0, 2, 7]
    >>> values, counts = unique_counts(x)
    >>> values
    (0, 2, 3, 4, 7, 8)
    >>> counts
    (5, 2, 2, 1, 3, 1)
    """
    xv, xc = zip(*[(k, len(list(g))) for k, g in groupby(sorted(x))])
    return xv, xc
