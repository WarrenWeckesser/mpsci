"""
Beta prime probability distribution
-----------------------------------

See https://en.wikipedia.org/wiki/Beta_prime_distribution

"""
from mpmath import mp
from ._common import (_validate_p, _validate_moment_n, _find_bracket,
                      _validate_x_bounds, Initial)
from ..stats import mean as _mean
from .. import fun as _fun


__all__ = ['pdf', 'logpdf', 'cdf', 'invcdf', 'sf', 'invsf',
           'mean', 'mode', 'var', 'skewness',
           'noncentral_moment']


def _validate_a_b(a, b):
    if a <= 0 or b <= 0:
        raise ValueError('The shape parameters a and b must be greater '
                         'than 0.')
    a = mp.mpf(a)
    b = mp.mpf(b)
    return a, b


def pdf(x, a, b):
    """
    Probability density function (PDF) for the beta prime distribution.
    """
    with mp.extradps(5):
        a, b = _validate_a_b(a, b)
        x = mp.mpf(x)
        if x < 0:
            return mp.zero
        if x == 0 and a < 1:
            return mp.inf
        return (mp.power(x, a - 1) / mp.power(1 + x, a + b) /
                mp.beta(a, b))


def logpdf(x, a, b):
    """
    Natural logarithm of the PDF of the beta prime distribution.
    """
    with mp.extradps(5):
        a, b = _validate_a_b(a, b)
        x = mp.mpf(x)
        if x < 0:
            return mp.ninf
        if x == 0 and a < 1:
            return mp.inf
        return (_fun.xlogy(a - 1, x) - _fun.xlog1py(a + b, x)
                - _fun.logbeta(a, b))


def cdf(x, a, b):
    """
    Cumulative distribution function of the beta prime distribution.
    """
    with mp.extradps(5):
        a, b = _validate_a_b(a, b)
        x = mp.mpf(x)
        if x < 0:
            return mp.zero
        if x == mp.inf:
            return mp.one
        if x > 1:
            c = mp.betainc(b, a, x1=1/(1+x), x2=1, regularized=True)
        else:
            c = mp.betainc(a, b, x1=0, x2=x/(1+x), regularized=True)
        return c


def sf(x, a, b):
    """
    Survival function of the beta prime distribution.
    """
    with mp.extradps(5):
        a, b = _validate_a_b(a, b)
        x = mp.mpf(x)
        if x < 0:
            return mp.one
        if x == mp.inf:
            return mp.zero
        if x > 1:
            c = mp.betainc(b, a, x1=0, x2=1/(1+x), regularized=True)
        else:
            c = mp.betainc(a, b, x1=x/(1+x), x2=1, regularized=True)
        return c


def invcdf(p, a, b):
    """
    Inverse of the CDF of the beta prime distribution.
    """
    with mp.extradps(5):
        a, b = _validate_a_b(a, b)
        p = _validate_p(p)
        if p == 0:
            return mp.zero
        if p == 1:
            return mp.inf

        x0, x1 = _find_bracket(lambda x: cdf(x, a, b), p, 0, mp.inf)
        if x0 == x1:
            return x0
        x = mp.findroot(lambda x: cdf(x, a, b) - p, x0=(x0, x1),
                        solver='secant')
        return x


def invsf(p, a, b):
    """
    Inverse of the survival function of the beta prime distribution.
    """
    with mp.extradps(5):
        a, b = _validate_a_b(a, b)
        p = _validate_p(p)
        if p == 0:
            return mp.one
        if p == 1:
            return mp.zero

        x0, x1 = _find_bracket(lambda x: sf(x, a, b), p, 0, mp.inf)
        if x0 == x1:
            return x0
        x = mp.findroot(lambda x: sf(x, a, b) - p, x0=(x0, x1),
                        solver='secant')
        return x


def mean(a, b):
    """
    Mean of the beta prime distribution.
    """
    with mp.extradps(5):
        a, b = _validate_a_b(a, b)
        if b <= 1:
            return mp.inf
        return a / (b - 1)


def mode(a, b):
    """
    Mode of the beta prime distribution.
    """
    with mp.extradps(5):
        a, b = _validate_a_b(a, b)
        if a < 1:
            return mp.zero
        return (a - 1) / (b + 1)


def var(a, b):
    """
    Variance of the beta prime distribution.
    """
    with mp.extradps(5):
        a, b = _validate_a_b(a, b)
        if b <= 2:
            return mp.nan
        return (a * (a + b - 1)) / ((b - 2)*(b - 1)**2)


def skewness(a, b):
    """
    Skewness of the beta prime distribution.
    """
    with mp.extradps(5):
        a, b = _validate_a_b(a, b)
        if b <= 3:
            return mp.nan
        t1 = 2 * (2*a + b - 1) / (b - 3)
        t2 = mp.sqrt((b - 2) / (a*(a + b - 1)))
        return t1 * t2


def noncentral_moment(n, a, b):
    """
    Noncentral moment (i.e. raw moment) of the beta prime distribution.

    ``n`` is the order of the moment.  The moment is not defined if
    ``n >= b``; ``nan`` is returned in that case.
    """
    with mp.extradps(5):
        n = _validate_moment_n(n)
        a, b = _validate_a_b(a, b)
        if n >= b:
            return mp.nan
        return mp.beta(a + n, b - n) / mp.beta(a, b)


def nll(x, a, b):
    """
    Negative log-likelihood of the beta prime distribution.

    `x` must be a sequence of nonnegative numbers.
    """
    with mp.extradps(5):
        a, b = _validate_a_b(a, b)
        x = _validate_x_bounds(x, low=0, high=mp.inf,
                               strict_low=False, strict_high=True)
        return -mp.fsum([logpdf(t, a, b) for t in x])


def _mle_eqn_a_b(a, b, meanlogx, meanlog1px):
    d1 = mp.digamma(a + b)
    eq1 = mp.digamma(a) - d1 - (meanlogx - meanlog1px)
    eq2 = mp.digamma(b) - d1 + meanlog1px
    return eq1, eq2


def mle(x, *, a=None, b=None):
    """
    Maximum likelihood estimate for the beta prime distribution.

    `x` must be a sequence of nonnegative numbers.

    Returns (a, b), the maximum likelihood estimate.

    A numerical equation solver (`mpmath.mp.findroot`) is used to solve
    for the maximum likelihood estimate.  For some data, this solver may
    fail to converge.  If that happens, different initial guesses for the
    parameter values may be given by passing instances of the
    `mpsci.distributions.Initial` class for `a` or `b`, e.g.

        from mpsci.distributions import Initial
        ahat, bhat = mle(x, b=Initial(12))

    The default initial guesses for `a` and `b` are 1.
    """
    with mp.extradps(5):
        x = _validate_x_bounds(x, low=0, high=mp.inf,
                               strict_low=False, strict_high=True)
        meanlog1px = _mean([mp.log1p(t) for t in x])
        meanlogx = _mean([mp.log(t) for t in x])

        def eqns(a, b):
            return _mle_eqn_a_b(a, b, meanlogx, meanlog1px)

        if ((a is None or isinstance(a, Initial))
                and (b is None or isinstance(b, Initial))):
            a0 = 1 if a is None else a.initial
            b0 = 1 if b is None else b.initial
            ahat, bhat = mp.findroot(eqns, [a0, b0])
            return ahat, bhat
        raise NotImplementedError('Only implemented for both a and b free '
                                  'parameters')
