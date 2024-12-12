"""
Binomial distribution
---------------------

"""

from mpmath import mp
from ..stats import mean as _mean, var as _var
from ._common import (_validate_p, _validate_x_bounds, _validate_counts,
                      isfixed)
from ..fun import logbinomial


__all__ = ['support', 'pmf', 'logpmf', 'cdf', 'sf', 'mean', 'var',
           'nll', 'mle']


def _wvar(x, weights=None, ddof=0):
    if weights is None:
        return _var(x, ddof=ddof)
    x = [mp.mpf(t) for t in x]
    weights = [mp.mpf(t) for t in weights]
    mu = _mean(x, weights=weights)
    v = _mean([(t - mu)**2 for t in x], weights=weights)
    if ddof != 0:
        n = len(x)
        v = n*v/(n - ddof)
    return v


def _validate_np(n, p):
    p = _validate_p(p)
    if n < 0 or int(n) != n:
        raise ValueError('n must be a nonnegative integer.')
    return n, p


def support(n, p):
    """
    Support of the binomial distribution.

    The support is the integers 0, 1, 2, ..., n; this is implemented
    by returning `range(n + 1)`.  That is, the return value is the
    `range` instance, not a sequence.

    Examples
    --------
    >>> from mpsci.distributions import binomial
    >>> sup = binomial.support(5, 0.25)
    >>> sup
    range(0, 6)
    >>> list(sup)
    [0, 1, 2, 3, 4, 5]

    """
    n, p = _validate_np(n, p)
    return range(n + 1)


def pmf(k, n, p):
    """
    Probability mass function of the binomial distribution.
    """
    with mp.extradps(5):
        n, p = _validate_np(n, p)
        return (mp.binomial(n, k) *
                mp.power(p, k) *
                mp.power(1 - p, n - k))


def logpmf(k, n, p):
    """
    Natural log of the probability mass function of the binomial distribution.
    """
    with mp.extradps(5):
        n, p = _validate_np(n, p)
        return (logbinomial(n, k)
                + k*mp.log(p)
                + mp.fsum([n, -k])*mp.log1p(-p))


def cdf(k, n, p, method='incbeta'):
    """
    Cumulative distribution function of the binomial distribution.

    `method` must be either "sumpmf" or "incbeta".  When `method` is "sumpmf",
    the CDF is computed with a simple sum of the PMF values.  When `method`
    is "incbeta", the incomplete beta function is used. This method is
    generally faster than the "sumpmf" method, but for large values of k
    or n, the incomplete beta function of mpmath might fail.
    """
    if method not in ['sumpmf', 'incbeta']:
        raise ValueError('method must be "sumpmf" or "incbeta"')
    if method == 'incbeta':
        with mp.extradps(5):
            n, p = _validate_np(n, p)
            if k == n:
                return mp.one
            # XXX For large values of k and/or n, betainc fails. The failure
            # occurs in one of the hypergeometric functions.
            return mp.betainc(n - k, k + 1, x1=0, x2=1 - p,
                              regularized=True)
    else:
        # method is "sumpmf"
        with mp.extradps(5):
            n, p = _validate_np(n, p)
            c = mp.fsum([mp.exp(logpmf(t, n, p))
                         for t in range(k + 1)])
            return c


def sf(k, n, p, method='incbeta'):
    """
    Survival function of the binomial distribution.

    `method` must be either "sumpmf" or "incbeta".  When `method` is "sumpmf",
    the survival function is computed with a simple sum of the PMF values.
    When `method` is "incbeta", the incomplete beta function is used. This
    method is generally faster than the "sumpmf" method, but for large values
    of k or n, the incomplete beta function of mpmath might fail.
    """
    if method not in ['sumpmf', 'incbeta']:
        raise ValueError('method must be "sumpmf" or "incbeta"')
    if method == 'incbeta':
        with mp.extradps(5):
            n, p = _validate_np(n, p)
            # XXX For large values of k and/or n, betainc fails. The failure
            # occurs in one of the hypergeometric functions.
            return mp.betainc(n - k, k + 1, x1=1-p, x2=1,
                              regularized=True)
    else:
        # method is "sumpmf"
        with mp.extradps(5):
            n, p = _validate_np(n, p)
            c = mp.fsum([mp.exp(logpmf(t, n, p))
                         for t in range(k + 1, n + 1)])
            return c


def mean(n, p):
    """
    Mean of the binomial distribution.
    """
    with mp.extradps(5):
        n, p = _validate_np(n, p)
        return n*p


def var(n, p):
    """
    Variance of the binomial distribution.
    """
    with mp.extradps(5):
        n, p = _validate_np(n, p)
        return n * p * (1 - p)


def nll(x, n, p, *, counts=None):
    """
    Negative log-likelihood of the binomial distribution.

    `x` must be a sequence of nonnegative integers, with
    ``0 <= x[i] <= n``.
    """
    with mp.extradps(5):
        n, p = _validate_np(n, p)
        x = _validate_x_bounds(x, low=0, high=n,
                               strict_low=False, strict_high=False)
        if not all([mp.isint(t) for t in x]):
            raise ValueError('all values in x must be integers')
        counts = _validate_counts(x, counts, expand_none=True)
        return -mp.fsum([count*logpmf(t, n, p)
                         for t, count in zip(x, counts)])


def _p_threshold(k):
    kmax = max(k)
    d = _mean([mp.digamma(kmax + 1) - mp.digamma(kmax - ki + 1) for ki in k])
    return -mp.expm1(-d)


def mle(x, *, counts=None, n=None, p=None):
    """
    Maximum likelihood estimation for the binomial distribution.

    x  must be a sequence of nonnegative integers.

    Returns `n` (an integer) and `p` (a probability).

    *Note*

    When the parameter `n` is not fixed, the robustness of the numerical
    solution depends on the input data `x`.  For some `x`, the solver is
    likely to fail.

    Examples
    --------
    >>> from mpmath import mp
    >>> from mpsci.distributions import binomial

    >>> mp.dps = 50
    >>> x = [7, 9, 10, 10, 11, 12, 12, 13, 13, 14, 15, 15]
    >>> n, p = binomial.mle(x, n=16)
    >>> p
    mpf('0.734375')

    """
    with mp.extradps(5):
        all_int = all([int(xi) == xi for xi in x])
        if not all_int:
            raise ValueError('all values in x must be integers')
        x = _validate_x_bounds(x, low=0, high=mp.inf,
                               strict_low=False)
        xmax = max(x)
        counts = _validate_counts(x, counts, expand_none=False)
        n_fixed = isfixed(n)
        p_fixed = isfixed(p)
        if n_fixed:
            if xmax > n:
                raise ValueError(f'The fixed value of n ({n}) must not be '
                                 f'less than the maximum value in x ({xmax})')
            if p_fixed:
                # Other than validation, there is nothing to do.
                n, p = _validate_np(n, p)
                return n, p
            else:
                # XXX p=Initial(...) is ignored, because we have an explicit
                # formula. Should Initial be disallowed for p in this case?
                n, _ = _validate_np(n, 0.5)
                m = _mean(x, weights=counts)
                phat = m/n
                return int(n), phat
        # n is not fixed.
        if not p_fixed:
            if n is not None and n.initial < xmax:
                raise ValueError(f"Initial guess for n ({n.initial}) must not "
                                 f"be less than max(x) ({xmax}))")
            m = _mean(x, weights=counts)

            def mle_n_eqn(n):
                p1 = m/n
                return (-_mean([mp.digamma(n - xi + 1) for xi in x],
                               weights=counts)
                        + mp.digamma(n + 1) + mp.log1p(-p1))

            if n is None:
                v = _wvar(x, weights=counts)
                if (m - v) > 0:
                    n0 = m**2/(m - v)
                    if n0 < xmax:
                        n0 = xmax
                else:
                    n0 = xmax
            else:
                n0 = n.initial
            # print(f"{n0 = }")
            nhat = mp.findroot(mle_n_eqn, n0)
            if nhat < xmax:
                nhat = xmax
                return mle(x, counts=counts, n=nhat, p=p)
            phat = m/nhat
            # Integer n is required.
            if int(nhat) == nhat:
                return int(nhat), phat
            nhat0 = int(mp.floor(nhat))
            nhat1 = int(mp.ceil(nhat))
            phat0 = m/(nhat0)
            phat1 = m/(nhat1)
            nll0 = nll(x, n=nhat0, p=phat0, counts=counts)
            nll1 = nll(x, n=nhat1, p=phat1, counts=counts)
            if nll0 <= nll1:
                return nhat0, phat0
            else:
                return nhat1, phat1
        else:
            # n is free, p is fixed.
            _, p = _validate_np(1, p)
            pt = _p_threshold(x)
            if p >= pt:
                nhat = xmax
                return int(xmax), p
            m = _mean(x, weights=counts)

            def mle_n_eqn(n):
                return (-_mean([mp.digamma(n - xi + 1) for xi in x],
                               weights=counts)
                        + mp.digamma(n + 1) + mp.log1p(-p))

            n0 = xmax + 1 if n is None else n.initial
            try:
                nhat = mp.findroot(mle_n_eqn, n0)
            except ValueError:
                # Try again with a different initial condition.
                n0 = m/p
                nhat = mp.findroot(mle_n_eqn, n0)
            if nhat < xmax:
                nhat = xmax
                return int(nhat), p
            # Integer n is required.
            if int(nhat) == nhat:
                return int(nhat), p
            nhat0 = int(mp.floor(nhat))
            nhat1 = nhat0 + 1
            nll0 = nll(x, n=nhat0, p=p, counts=counts)
            nll1 = nll(x, n=nhat1, p=p, counts=counts)
            if nll0 <= nll1:
                return nhat0, p
            else:
                return nhat1, p
