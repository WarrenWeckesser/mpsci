
from dataclasses import dataclass
import operator
from mpmath import mp


@dataclass
class Initial:
    """
    This class allows an initial guess of a parameter to be passed to
    the `mle` function of a probability distribution.
    """

    # The value of the initial guess.
    initial: mp.mpf


def isfixed(param):
    return not (param is None or isinstance(param, Initial))


def _validate_loc_scale(loc, scale, scale_name='scale'):
    loc = mp.mpf(loc)
    scale = mp.mpf(scale)
    if scale <= 0:
        raise ValueError('f{scale_name} must be positive.')
    return loc, scale


def _validate_p(p):
    if p < 0 or p > 1:
        raise ValueError('p must be in the interval [0, 1]')
    return mp.mpf(p)


def _validate_moment_n(n):
    try:
        n = operator.index(n)
    except TypeError:
        raise TypeError('n must be an integer')
    if n < 0:
        raise ValueError('n must be nonnegative')
    return n


def _seq_to_mp(x):
    """
    Convert a 1D sequence of values in x to a sequence of mpmath numerial
    values.  ``mp.mpmathify`` can handle the various integer and float
    types in NumPy, and objects such as ``fractions.Fraction``.
    """
    return [mp.mpmathify(t) for t in x]


def _validate_x_bounds(x, low=None, high=None,
                       strict_low=False, strict_high=False,
                       lowname=None, highname=None):
    """
    Verify that all values in the sequence ``x`` are within the given bounds.

    If ``strict_low`` is True, each value must be strictly greater than
    ``low``.  Similarly, if ``strict_high`` is True, each value must be
    strictly less than ``high``.
    """
    if low is not None:
        if strict_low:
            bad = any(t <= low for t in x)
        else:
            bad = any(t < low for t in x)
        if bad:
            t1 = "or equal to " if not strict_low else ""
            if lowname is None:
                t2 = f'{low}'
            else:
                t2 = f'{lowname} ({low})'
            raise ValueError(f'All values in x must be greater than {t1}{t2}.')
    if high is not None:
        if strict_high:
            bad = any(t >= high for t in x)
        else:
            bad = any(t > high for t in x)
        if bad:
            t1 = "or equal to " if not strict_high else ""
            if highname is None:
                t2 = f'{high}'
            else:
                t2 = f'{highname} ({high})'
            raise ValueError(f'All values in x must be less than {t1}{t2}.')
    return _seq_to_mp(x)


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
        nbisect = 8
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
