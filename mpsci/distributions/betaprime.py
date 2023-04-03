"""
Beta prime probability distribution
-----------------------------------

See https://en.wikipedia.org/wiki/Beta_prime_distribution

The functions defined here include a scale parameter, so according to the
Wikipedia article, this is actually a generalization of the beta prime
distribution known as the *compound gamma distribution*.  If you want
the "standard" beta prime distribution as described in the article, set
`scale` to 1.
"""
from mpmath import mp
from ._common import (_validate_p, _validate_moment_n, _find_bracket,
                      _validate_x_bounds, Initial)
from ..stats import mean as _mean
from .. import fun as _fun


__all__ = ['pdf', 'logpdf', 'cdf', 'invcdf', 'sf', 'invsf',
           'mean', 'mode', 'var', 'skewness', 'noncentral_moment',
           'nll', 'mle']


def _validate_params(a, b, scale):
    if a <= 0 or b <= 0:
        raise ValueError('The shape parameters a and b must be greater '
                         'than 0.')
    if scale <= 0:
        raise ValueError('scale must be greater than 0.')
    a = mp.mpf(a)
    b = mp.mpf(b)
    scale = mp.mpf(scale)
    return a, b, scale


def pdf(x, a, b, scale):
    """
    Probability density function (PDF) for the beta prime distribution.
    """
    with mp.extradps(5):
        a, b, scale = _validate_params(a, b, scale)
        x = mp.mpf(x)
        if x < 0:
            return mp.zero
        if x == 0 and a < 1:
            return mp.inf
        z = x/scale
        return (mp.power(z, a - 1) / mp.power(1 + z, a + b) /
                mp.beta(a, b))/scale


def logpdf(x, a, b, scale):
    """
    Natural logarithm of the PDF of the beta prime distribution.
    """
    with mp.extradps(5):
        a, b, scale = _validate_params(a, b, scale)
        x = mp.mpf(x)
        if x < 0:
            return mp.ninf
        if x == 0 and a < 1:
            return mp.inf
        z = x/scale
        return (_fun.xlogy(a - 1, z) - _fun.xlog1py(a + b, z)
                - _fun.logbeta(a, b) - mp.log(scale))


def cdf(x, a, b, scale):
    """
    Cumulative distribution function of the beta prime distribution.
    """
    with mp.extradps(5):
        a, b, scale = _validate_params(a, b, scale)
        x = mp.mpf(x)
        if x < 0:
            return mp.zero
        if x == mp.inf:
            return mp.one
        z = x/scale
        if z > 1:
            c = mp.betainc(b, a, x1=1/(1+z), x2=1, regularized=True)
        else:
            c = mp.betainc(a, b, x1=0, x2=z/(1+z), regularized=True)
        return c


def sf(x, a, b, scale):
    """
    Survival function of the beta prime distribution.
    """
    with mp.extradps(5):
        a, b, scale = _validate_params(a, b, scale)
        x = mp.mpf(x)
        if x < 0:
            return mp.one
        if x == mp.inf:
            return mp.zero
        z = x/scale
        if z > 1:
            c = mp.betainc(b, a, x1=0, x2=1/(1+z), regularized=True)
        else:
            c = mp.betainc(a, b, x1=z/(1+z), x2=1, regularized=True)
        return c


def invcdf(p, a, b, scale):
    """
    Inverse of the CDF of the beta prime distribution.
    """
    with mp.extradps(5):
        a, b, scale = _validate_params(a, b, scale)
        p = _validate_p(p)
        if p == 0:
            return mp.zero
        if p == 1:
            return mp.inf

        x0, x1 = _find_bracket(lambda x: cdf(x, a, b, scale), p, 0, mp.inf)
        if x0 == x1:
            return x0
        x = mp.findroot(lambda x: cdf(x, a, b, scale) - p, x0=(x0, x1),
                        solver='secant')
        return x


def invsf(p, a, b, scale):
    """
    Inverse of the survival function of the beta prime distribution.
    """
    with mp.extradps(5):
        a, b, scale = _validate_params(a, b, scale)
        p = _validate_p(p)
        if p == 0:
            return mp.one
        if p == 1:
            return mp.zero

        x0, x1 = _find_bracket(lambda x: sf(x, a, b, scale), p, 0, mp.inf)
        if x0 == x1:
            return x0
        x = mp.findroot(lambda x: sf(x, a, b, scale) - p, x0=(x0, x1),
                        solver='secant')
        return x


def mean(a, b, scale):
    """
    Mean of the beta prime distribution.
    """
    with mp.extradps(5):
        a, b, scale = _validate_params(a, b, scale)
        if b <= 1:
            return mp.inf
        return scale*a/(b - 1)


def mode(a, b, scale):
    """
    Mode of the beta prime distribution.
    """
    with mp.extradps(5):
        a, b, scale = _validate_params(a, b, scale)
        if a < 1:
            return mp.zero
        return scale * (a - 1) / (b + 1)


def var(a, b, scale):
    """
    Variance of the beta prime distribution.
    """
    with mp.extradps(5):
        a, b, scale = _validate_params(a, b, scale)
        if b <= 2:
            return mp.nan
        return scale**2 * (a * (a + b - 1)) / ((b - 2)*(b - 1)**2)


def skewness(a, b, scale):
    """
    Skewness of the beta prime distribution.
    """
    with mp.extradps(5):
        a, b, scale = _validate_params(a, b, scale)
        if b <= 3:
            return mp.nan
        t1 = 2 * (2*a + b - 1) / (b - 3)
        t2 = mp.sqrt((b - 2) / (a*(a + b - 1)))
        return t1 * t2


def noncentral_moment(n, a, b, scale):
    """
    Noncentral moment (i.e. raw moment) of the beta prime distribution.

    ``n`` is the order of the moment.  The moment is not defined if
    ``n >= b``; ``nan`` is returned in that case.
    """
    with mp.extradps(5):
        n = _validate_moment_n(n)
        a, b, scale = _validate_params(a, b, scale)
        if n >= b:
            return mp.nan
        return scale**n * mp.beta(a + n, b - n) / mp.beta(a, b)


def nll(x, a, b, scale):
    """
    Negative log-likelihood of the beta prime distribution.

    `x` must be a sequence of nonnegative numbers.
    """
    with mp.extradps(5):
        a, b, scale = _validate_params(a, b, scale)
        x = _validate_x_bounds(x, low=0, high=mp.inf,
                               strict_low=False, strict_high=True)
        return -mp.fsum([logpdf(t, a, b, scale) for t in x])


def _is_fixed(param):
    return param is not None and not isinstance(param, Initial)


def mle(x, *, a=None, b=None, scale=None):
    """
    Maximum likelihood estimate for the beta prime distribution.

    `x` must be a sequence of nonnegative numbers.

    Returns (a, b, scale), the maximum likelihood estimate.

    A numerical equation solver (`mpmath.mp.findroot`) is used to solve
    for the maximum likelihood estimate.  For some data, this solver may
    fail to converge.  If that happens, different initial guesses for the
    parameter values may be given by passing instances of the
    `mpsci.distributions.Initial` class for `a` or `b`, e.g.

        from mpsci.distributions import Initial
        ahat, bhat, scalehat = mle(x, b=Initial(12))

    The default initial guess for all the parameters is 1.
    """
    a_fixed = _is_fixed(a)
    b_fixed = _is_fixed(b)
    scale_fixed = _is_fixed(scale)

    with mp.extradps(5):
        x = _validate_x_bounds(x, low=0, high=mp.inf,
                               strict_low=False, strict_high=True)

        if a_fixed and b_fixed and scale_fixed:
            # All parameters fixed--nothing to do.
            return _validate_params(a, b, scale)

        if not a_fixed and not b_fixed and not scale_fixed:
            # All parameters are free.
            meanlogx = _mean([mp.log(t) for t in x])

            def eqns_a_b_scale(a, b, scale):
                meanlog1px = _mean([mp.log1p(t/scale) for t in x])
                meanxover1px = _mean([(t/scale)/(1 + t/scale) for t in x])
                d1 = mp.digamma(a + b)
                eq1 = (mp.digamma(a) - d1 - (meanlogx - meanlog1px)
                       + mp.log(scale))
                eq2 = mp.digamma(b) - d1 + meanlog1px
                eq3 = a - (a + b)*meanxover1px
                return eq1, eq2, eq3

            a0 = mp.mpf(1 if a is None else a.initial)
            b0 = mp.mpf(1 if b is None else b.initial)
            scale0 = mp.mpf(1 if scale is None else scale.initial)

            ahat, bhat, scalehat = mp.findroot(eqns_a_b_scale,
                                               [a0, b0, scale0],
                                               solver='secant')
            return ahat, bhat, scalehat

        if scale_fixed:
            if not (a_fixed or b_fixed):
                # scale is fixed, a and b are free.
                _, _, scale = _validate_params(1, 1, scale)
                meanlog1px = _mean([mp.log1p(t/scale) for t in x])
                meanlogx = _mean([mp.log(t) for t in x])
                lnscale = mp.log(scale)

                def eqns_a_b(a, b):
                    d1 = mp.digamma(a + b)
                    eq1 = (mp.digamma(a) - d1 - (meanlogx - meanlog1px)
                           + lnscale)
                    eq2 = mp.digamma(b) - d1 + meanlog1px
                    return eq1, eq2

                a0 = 1 if a is None else a.initial
                b0 = 1 if b is None else b.initial
                ahat, bhat = mp.findroot(eqns_a_b, [a0, b0])
                return ahat, bhat, scale
            elif a_fixed and not b_fixed:
                a, _, scale = _validate_params(a, 1, scale)
                meanlog1px = _mean([mp.log1p(t/scale) for t in x])

                def b_eqn(b):
                    return mp.digamma(b) - mp.digamma(a + b) + meanlog1px

                b0 = mp.mpf(1 if b is None else b.initial)
                bhat = mp.findroot(b_eqn, b0)
                return a, bhat, scale
            else:
                # scale and b are fixed, a is not fixed.
                _, b, scale = _validate_params(1, b, scale)
                meanlog1px = _mean([mp.log1p(t/scale) for t in x])
                meanlogx = _mean([mp.log(t) for t in x])
                logscale = mp.log(scale)

                def a_eqn(a):
                    return (mp.digamma(a) - mp.digamma(a + b)
                            - meanlogx + meanlog1px + logscale)

                a0 = mp.mpf(1 if a is None else a.initial)
                ahat = mp.findroot(a_eqn, a0)
                return ahat, b, scale
        else:
            # scale is not fixed, but one of a or b is fixed.
            if a_fixed and b_fixed:
                a, b, _ = _validate_params(a, b, 1)
                scale0 = mp.mpf(1 if scale is None else scale.initial)

                def scale_eqn(s):
                    return a - (a + b)*_mean([(xi/s)/(1 + xi/s) for xi in x])

                scalehat = mp.findroot(scale_eqn, scale0)
                return a, b, scalehat
            elif a_fixed and not b_fixed:
                a, _, _ = _validate_params(a, 1, 1)
                b0 = mp.mpf(1 if b is None else b.initial)
                scale0 = mp.mpf(1 if scale is None else scale.initial)

                def scale_eqn(s):
                    meanxover1px = _mean([(t/s)/(1 + t/s) for t in x])
                    meanlog1px = _mean([mp.log1p(t/s) for t in x])
                    return (mp.digamma(a*(1/meanxover1px - 1))
                            - mp.digamma(a/meanxover1px)
                            + meanlog1px)

                scalehat = mp.findroot(scale_eqn, scale0)
                meanxover1px = _mean([(t/scalehat)/(1 + t/scalehat)
                                      for t in x])
                bhat = a * (1/meanxover1px - 1)
                return a, bhat, scalehat
            else:
                # scale and a are not fixed, b is fixed.
                _, b, _ = _validate_params(1, b, 1)
                a0 = mp.mpf(1 if a is None else a.initial)
                scale0 = mp.mpf(1 if scale is None else scale.initial)

                def scale_eqn(s):
                    meanxover1px = _mean([(t/s)/(1 + t/s) for t in x])
                    meanlog1px = _mean([mp.log1p(t/s) for t in x])
                    meanlogx = _mean([mp.log(t) for t in x])
                    return (mp.digamma(b*(meanxover1px/(1 - meanxover1px)))
                            - mp.digamma(b/(1 - meanxover1px))
                            - meanlogx
                            + meanlog1px
                            + mp.log(s))

                scalehat = mp.findroot(scale_eqn, scale0)
                meanxover1px = _mean([(t/scalehat)/(1 + t/scalehat)
                                      for t in x])
                ahat = b * (meanxover1px/(1 - meanxover1px))
                return ahat, b, scalehat
