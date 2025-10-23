
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


def _validate_int(k):
    """
    Verify that the value of k is integral, and return int(k).
    """
    if k != int(k):
        raise ValueError('k must be an integer')
    return int(k)


def _validate_moment_n(n):
    try:
        n = operator.index(n)
    except TypeError:
        raise TypeError('n must be an integer') from None
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


def _validate_counts(x, counts, expand_none=True):
    n = len(x)
    if counts is None:
        if expand_none:
            counts = [1]*n
        return counts
    if len(counts) != n:
        raise ValueError('len(counts) must equal len(x); '
                         f'got {len(counts)=} and {len(x)=}')
    if any([t != int(t) or t < 0 for t in counts]):
        raise ValueError('counts must contain only nonnegative integers')
    return counts


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
    for _k in range(nbisect):
        mid = (x0 + x1)/2
        pmid = func(mid)
        if pmid == p:
            return (mid, mid)
        if mp.sign(p - pmid) == sign:
            x0 = mid
        else:
            x1 = mid

    return x0, x1


def _find_bracket_by_expansion_neginf_inf(func, p, direction):
    """
    Find an interval [x_low, x_high] that contains the solution to func(x) = p.

    func must be strictly monotonic on the interval (-inf, inf).

    direction = 1:  func is increasing
    direction = -1: func is decreasing
    """
    # The initial guess for x_low and x_high is 0.  If 0 doesn't provide
    # a bound, the magnitude of the next guess is zero_step.  From then
    # on, the guess is multiplied by 1.5 until a bound is found.
    zero_step = mp.one
    if direction not in [-1, 1]:
        raise ValueError('direction must be -1 or 1.')
    if direction == 1:
        not_high_enough = lambda x: func(x) < p
        not_low_enough = lambda x: func(x) > p
    else:
        not_high_enough = lambda x: func(x) > p
        not_low_enough = lambda x: func(x) < p

    x_high = mp.zero
    if not_high_enough(x_high):
        x_high = zero_step
        while not_high_enough(x_high):
            x_high *= 1.5

    x_low = mp.zero
    if not_low_enough(x_low):
        x_low = -zero_step
        while not_low_enough(x_low):
            x_low *= 1.5

    return (x_low, x_high)


def _find_bracket_by_expansion_0_inf(func, p, direction):
    """
    Find an interval [x_low, x_high] that contains the solution to func(x) = p.

    func must be strictly monotonic on the interval [0, inf).

    direction = 1:  func is increasing
    direction = -1: func is decreasing
    """
    if direction not in [-1, 1]:
        raise ValueError('direction must be -1 or 1.')
    if direction == 1:
        not_high_enough = lambda x: func(x) < p
        not_low_enough = lambda x: func(x) > p
    else:
        not_high_enough = lambda x: func(x) > p
        not_low_enough = lambda x: func(x) < p
    x_high = mp.one
    while not_high_enough(x_high):
        x_high *= 1.5
    # XXX/FIXME: The following assumes there is an x in (0, 1]
    # where not_low_enough(x) will be false.
    x_low = mp.one
    while x_low > 0 and not_low_enough(x_low):
        x_low *= 0.5
    return (x_low, x_high)


def _generic_inv(func, p, direction, solver='bisect', **kwargs):
    """
    Invert a strictly monotonic function whose domain is (0, inf).

    That is, solve func(x) = p for x, assuming 0 < x < inf, and
    func is strictly monotonic.

    direction = 1:  func is increasing
    direction = -1: func is decreasing

    Additional keyword arguments are passed on to `mp.findroot()`.

    If not given in `kwargs`, this function overrides the default
    `maxsteps` of `mp.findroot` and uses (in effect) 4*mp.prec.

    Experimental!
    This function has only been used with functions that are positive, with
    range in [0, 1] (distribution CDF and SF functions).  It might work fine
    with a broader class of functions, but that is untested.
    Also, it has been developed mostly with `solver='bisect'`.
    The other bisection-based solvers should also work OK.
    The other classes of solvers (i.e. not bisection-based, such as 'secant'
    and 'halley') have not been tested, and might have convergence issues.
    """
    # Note: 2*mp.prec is probably overkill...
    with mp.workprec(2*mp.prec):
        maxsteps = kwargs.pop('maxsteps', 2*mp.prec)
        # Find a bracket for the root.
        k_low, k_high = _find_bracket_by_expansion_0_inf(func, p,
                                                         direction=direction)
        if solver in ['bisect', 'anderson', 'illinois', 'pegasus']:
            k0 = (k_low, k_high)
        else:
            k0 = (k_low + k_high)/2
        return mp.findroot(lambda k: func(k) - p,
                           x0=k0, solver=solver, maxsteps=maxsteps, **kwargs)
