
from mpmath import mp


def _validate_p(p):
    if p < 0 or p > 1:
        raise ValueError('p must be in the interval [0, 1]')
    return mp.mpf(p)


def _median(x):
    """
    Compute the median of the sequence x.
    """
    xs = sorted(x)
    n = len(xs)
    m = n // 2
    if n & 1:
        med = mp.mpf(xs[m])
    else:
        med = mp.fsum(xs[m - 1:m + 1])/2
    return med


def _find_bracket(func, p, a, b, nbisect=None):
    """
    Find an interval for solving func(x) = p.

    `a` and `b` must satisfy a < b.  `a` may be -inf, and
    b may be inf.

    `func` must be finite and strictly monotone on the interval [a, b].

    """
    p = mp.mpf(p)
    a = mp.mpf(a)
    b = mp.mpf(b)

    pa = func(a)
    if p == pa:
        return (a, a)
    pb = func(b)
    if p == pb:
        return (b, b)

    sign = mp.sign(pb - pa)

    if a == mp.ninf and b == mp.inf:
        x0 = -mp.one
        x1 = mp.one
        while mp.sign(p - func(x0)) != sign:
            x1 = x0
            x0 = 2*x0
        while mp.sign(p - func(x1)) == sign:
            x1 = 2*x1
    elif a == mp.ninf:
        delta = mp.one
        x1 = b
        x0 = b - delta
        while mp.sign(p - func(x0)) != sign:
            x1 = x0
            delta = 2*delta
            x0 = b - delta
    elif b == mp.inf:
        delta = mp.one
        x0 = a
        x1 = a + delta
        while mp.sign(p - func(x1)) == sign:
            x0 = x1
            delta = 2*delta
            x1 = a + delta
    else:
        # a and b are finite
        x0 = a
        x1 = b

    # [x0, x1] are now a bracket of the root.  Apply a few iterations
    # of bisection to refine.

    if nbisect is None:
        nbisect = max(mp.prec // 2, 8)
    for k in range(nbisect):
        mid = (x0 + x1)/2
        pmid = func(mid)
        if pmid == p:
            return (mid, mid)
        if mp.sign(p - pmid) == sign:
            x0 = mid
        else:
            x1 = mid

    return x0, x1
