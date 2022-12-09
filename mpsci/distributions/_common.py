
import mpmath


def _validate_p(p):
    if p < 0 or p > 1:
        raise ValueError('p must be in the interval [0, 1]')
    return mpmath.mp.mpf(p)


def _median(x):
    """
    Compute the median of the sequence x.
    """
    xs = sorted(x)
    n = len(xs)
    m = n // 2
    if n & 1:
        med = mpmath.mpf(xs[m])
    else:
        med = mpmath.fsum(xs[m - 1:m + 1])/2
    return med


def _get_interval_cdf(func, p):
    """
    Find an interval for solving func(x) = p.
    """
    x0 = mpmath.mp.one
    while func(x0) > p:
        x0 = 0.5*x0
    if func(x0) == p:
        return (x0, x0)
    while func(x0) < p:
        x0 = x0/0.875
    if func(x0) == p:
        return (x0, x0)
    x1 = x0
    x0 = 0.875*x0
    return x0, x1
