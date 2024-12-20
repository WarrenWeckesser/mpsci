"""
Beta probability distribution
-----------------------------

This is the "standard" beta distribution, with shape parameters
`a` and `b`, and support on the interval [0, 1].
"""

from mpmath import mp
from ._common import (_validate_p, _validate_moment_n, _validate_x_bounds,
                      _find_bracket, isfixed)
from .. import fun as _fun
from ..stats import mean as _mean, var as _var


__all__ = ['pdf', 'logpdf', 'cdf', 'invcdf', 'sf', 'invsf', 'interval_prob',
           'support',
           'mean', 'var', 'skewness', 'kurtosis', 'nll', 'noncentral_moment',
           'entropy', 'mle', 'mom']


def _validate_a_b(a, b):
    if a <= 0 or b <= 0:
        raise ValueError('The shape parameters a and b must be greater '
                         'than 0.')
    return mp.mpf(a), mp.mpf(b)


def pdf(x, a, b):
    """
    Probability density function (PDF) for the beta distribution.
    """
    with mp.extradps(5):
        a, b = _validate_a_b(a, b)
        x = mp.mpf(x)
        if x < 0 or x > 1:
            return mp.zero
        if x == 0 and a < 1:
            return mp.inf
        if x == 1 and b < 1:
            return mp.inf
        return (mp.power(x, a - 1) * mp.power(1 - x, b - 1) /
                mp.beta(a, b))


def logpdf(x, a, b):
    """
    Natural logarithm of the PDF of the beta distribution.
    """
    with mp.extradps(5):
        a, b = _validate_a_b(a, b)
        x = mp.mpf(x)
        if x < 0 or x > 1:
            return -mp.inf
        return (_fun.xlogy(a - 1, x) + _fun.xlog1py(b - 1, -x)
                - _fun.logbeta(a, b))


def cdf(x, a, b):
    """
    Cumulative distribution function of the beta distribution.
    """
    with mp.extradps(5):
        a, b = _validate_a_b(a, b)
        if x < 0:
            return mp.zero
        if x > 1:
            return mp.one
        return mp.betainc(a, b, x1=0, x2=x, regularized=True)


def sf(x, a, b):
    """
    Survival function of the beta distribution.
    """
    with mp.extradps(5):
        a, b = _validate_a_b(a, b)
        if x < 0:
            return mp.one
        if x > 1:
            return mp.zero
        return mp.betainc(a, b, x1=x, x2=1, regularized=True)


def interval_prob(x1, x2, a, b):
    """
    Compute the probability of x in [x1, x2] for the beta distribution.

    Mathematically, this is the same as

        beta.cdf(x2, a, b) - beta.cdf(x1, a, b)

    but when the two CDF values are nearly equal, this function will give
    a more accurate result.

    x1 must be less than or equal to x2.
    """
    with mp.extradps(5):
        a, b = _validate_a_b(a, b)
        x1 = mp.mpf(x1)
        x2 = mp.mpf(x2)
        if x1 > x2:
            raise ValueError('x1 must not be greater than x2')
        return mp.betainc(a, b, x1, x2, regularized=True)


def invcdf(p, a, b):
    """
    Inverse of the CDF of the beta distribution.
    """
    with mp.extradps(5):
        a, b = _validate_a_b(a, b)
        p = _validate_p(p)

        x0, x1 = _find_bracket(lambda x: cdf(x, a, b), p, 0, 1)
        if x0 == x1:
            return x0

        return _fun.betaincinv(a, b, p, method=('bisect', [x0, x1]))


def invsf(p, a, b):
    """
    Inverse of the survival function of the beta distribution.
    """
    with mp.extradps(5):
        a, b = _validate_a_b(a, b)
        p = _validate_p(p)

        x0, x1 = _find_bracket(lambda x: sf(x, a, b), p, 0, 1)
        if x0 == x1:
            return x0

        return _fun.betaincinv(a, b, p, complement=True,
                               method=('bisect', [x0, x1]))


def support(a, b):
    """
    Support of the beta distribution.
    """
    with mp.extradps(5):
        a, b = _validate_a_b(a, b)
        return (mp.zero, mp.one)


def mean(a, b):
    """
    Mean of the beta distribution.
    """
    with mp.extradps(5):
        a, b = _validate_a_b(a, b)
        return a/(a + b)


def var(a, b):
    """
    Variance of the beta distribution.
    """
    with mp.extradps(5):
        a, b = _validate_a_b(a, b)
        apb = a + b
        return a*b/(apb**2 * (apb + 1))


def skewness(a, b):
    """
    Skewness of the beta distribution.
    """
    with mp.extradps(5):
        a, b = _validate_a_b(a, b)
        apb = a + b
        return 2*(b - a)*mp.sqrt(apb + 1)/((apb + 2)*(mp.sqrt(a*b)))


def kurtosis(a, b):
    """
    Excess kurtosis of the beta distribution.
    """
    with mp.extradps(5):
        a, b = _validate_a_b(a, b)
        apb = a + b
        return (6*((a - b)**2*(apb + 1) - a*b*(apb + 2)) /
                (a*b*(apb + 2)*(apb + 3)))


def noncentral_moment(n, a, b):
    """
    n-th noncentral moment of the beta distribution.

    n must be a nonnegative integer.
    """
    n = _validate_moment_n(n)
    with mp.extradps(5):
        a, b = _validate_a_b(a, b)
        mu = mp.one
        for k in range(n):
            mu *= (a + k)/(a + b + k)
        return mu


def entropy(a, b):
    """
    Differential entropy of the beta distribution.
    """
    with mp.extradps(5):
        a, b = _validate_a_b(a, b)
        return (_fun.logbeta(a, b)
                - (a - 1)*mp.psi(0, a)
                - (b - 1)*mp.psi(0, b)
                + (a + b - 2)*mp.psi(0, a + b))


def nll(x, a, b):
    """
    Negative log-likelihood for the beta distribution.
    """
    x = _validate_x_bounds(x, low=0, high=1, strict_low=True, strict_high=True)
    return -mp.fsum([logpdf(t, a, b) for t in x])


def _beta_mle_func(a, b, n, s):
    psiab = mp.digamma(a + b)
    return s - n * (-psiab + mp.digamma(a))


def mle(x, *, a=None, b=None):
    """
    Maximum likelihood estimation for the beta distribution.
    """
    x = _validate_x_bounds(x, low=0, high=1, strict_low=True, strict_high=True)
    n = len(x)
    xbar = _mean(x)
    a_is_fixed = isfixed(a)
    if a_is_fixed:
        if a <= 0:
            raise ValueError('a must be greater than 0')
        else:
            a = mp.mpf(a)
    b_is_fixed = isfixed(b)
    if b_is_fixed:
        if b <= 0:
            raise ValueError('b must be greater than 0')
        else:
            b = mp.mpf(b)

    if not a_is_fixed and not b_is_fixed:
        # Fit both parameters
        s1 = mp.fsum(mp.log(t) for t in x)
        s2 = mp.fsum(mp.log1p(-t) for t in x)
        fac = xbar * (mp.one - xbar) / _var(x) - 1
        if a is None:
            a0 = xbar * fac
        else:
            # a must be an instance of Initial.
            a0 = a.initial
        if b is None:
            b0 = (mp.one - xbar) * fac
        else:
            # b must be an instance of Initial.
            b0 = b.initial
        a1, b1 = mp.findroot([lambda a, b: _beta_mle_func(a, b, n, s1),
                              lambda a, b: _beta_mle_func(b, a, n, s2)],
                             [a0, b0])
        return a1, b1

    if a_is_fixed and b_is_fixed:
        return a, b

    swap = False
    if a_is_fixed:
        swap = True
        b, a = a, b
        x = [mp.one - t for t in x]
        xbar = mp.one - xbar

    # Fit the a parameter, with b given.
    s1 = mp.fsum(mp.log(t) for t in x)
    if a is None:
        p0 = b * xbar / (mp.one - xbar)
    else:
        # a must be an instance of Initial.
        p0 = a.initial
    p1 = mp.findroot(lambda a: _beta_mle_func(a, b, n, s1), p0)
    if swap:
        return b, p1
    else:
        return p1, b


def mom(x):
    """
    Method of moments parameter estimation for the beta distribution.

    x must be a sequence of numbers from the interval (0, 1).

    Returns (a, b).
    """
    with mp.extradps(5):
        x = _validate_x_bounds(x, low=0, high=1,
                               strict_low=True, strict_high=True)
        M1 = _mean(x)
        M2 = _mean([t**2 for t in x])
        c = (M1 - M2) / (M2 - M1**2)
        a = M1*c
        b = (1 - M1)*c
        return a, b


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Experimental code -- fit the beta distribution to interval-censored
# data and compute the standard errors of the parameter estimates.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def _nll_cens(a, b, lo, hi):
    t = -sum(mp.log(interval_prob(x1, x2, a, b))
             for (x1, x2) in zip(lo, hi))
    return t


def _nll_cens_da(a, b, lo, hi):
    return mp.diff(lambda t: _nll_cens(t, b, lo, hi), a)


def _nll_cens_db(a, b, lo, hi):
    return mp.diff(lambda t: _nll_cens(a, t, lo, hi), b)


def _nll_cens_d2a(a, b, lo, hi):
    return mp.diff(lambda t: _nll_cens(t, b, lo, hi), a, n=2)


def _nll_cens_d2b(a, b, lo, hi):
    return mp.diff(lambda t: _nll_cens(a, t, lo, hi), b, n=2)


def _nll_cens_dadb(a, b, lo, hi):
    return mp.diff(lambda t: _nll_cens_da(a, t, lo, hi), b)


def _mle_cens(lo, hi, p0=None):
    """
    MLE for interval-censored data.

    Returns a, b, stderr(a), stderr(b).

    Examples
    --------
    >>> from mpmath import mp
    >>> from mpsci.distributions import beta
    >>> mp.dps = 60
    >>> lo = [0.0, 0.2, 0.4, 0.6, 0.8]
    >>> hi = [0.2, 0.4, 0.6, 0.8, 1.0]
    >>> counts = [5, 4, 7, 3, 1]
    >>> lower = sum(([v]*k for (v, k) in zip(lo, counts)), []) # like np.repeat
    >>> upper = sum(([v]*k for (v, k) in zip(hi, counts)), [])
    >>> a, b, hess, se_a, se_b = beta._mle_cens(lower, upper)
    >>> a, b
    (mpf('1.43063996669755867460949981031263342116947521525450502076755418'),
     mpf('2.10146607361141987108020246352181196280047393503978656215126088'))
    >>> se_a, se_b
    (mpf('0.560981296555925349878923728306227202047919510954353813044627044'),
     mpf('0.804302808645818037653239966888036072761898836526193514644697737'))

    Compare that to the result of the following R code::

        library(fitdistrplus)
        lo <- c(0. , 0. , 0. , 0. , 0. , 0.2, 0.2, 0.2, 0.2, 0.4, 0.4, 0.4,
                0.4, 0.4, 0.4, 0.4, 0.6, 0.6, 0.6, 0.8)
        hi <- c(0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.4, 0.4, 0.4, 0.6, 0.6, 0.6,
                0.6, 0.6, 0.6, 0.6, 0.8, 0.8, 0.8, 1. )
        data <- data.frame(left=lo, right=hi)
        result <- fitdistcens(data, 'beta', control=list(reltol=1e-13))
        print(result)
        cat("\nStandard deviations of parameter estimates:\n")
        print(result$sd)

    which prints (perhaps with slightly different line wrapping)::

        Fitting of the distribution ' beta ' on censored data by maximum
        likelihood Parameters:
               estimate
        shape1 1.430639
        shape2 2.101466

        Standard deviations of parameter estimates:
           shape1    shape2
        0.5609800 0.8043017

    """
    with mp.extradps(5):
        if p0 is None:
            mid = [0.5*(s + t) for (s, t) in zip(lo, hi)]
            p0 = mle(mid)
        a, b = mp.findroot([lambda a, b: _nll_cens_da(a, b, lo, hi),
                            lambda a, b: _nll_cens_db(a, b, lo, hi)],
                           p0)

        hess = mp.matrix(2)
        hess[0, 0] = _nll_cens_d2a(a, b, lo, hi)
        hess[1, 1] = _nll_cens_d2b(a, b, lo, hi)
        hess[0, 1] = _nll_cens_dadb(a, b, lo, hi)
        hess[1, 0] = hess[0, 1]

        hessinv = mp.inverse(hess)
        se_a = mp.sqrt(hessinv[0, 0])
        se_b = mp.sqrt(hessinv[1, 1])

        return a, b, hess, se_a, se_b
