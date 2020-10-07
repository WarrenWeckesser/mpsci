"""
Weibull probability distribution (for maxima)
---------------------------------------------

This is the same distribution as:

* `scipy.stats.weibull_max`;
* the Type III extreme value distribution used in the text "An Introduction
  to Statistical Modeling of Extreme Values" by Stuart Coles (Springer, 2001);
* the Weibull distribution given in the text "Modelling Extremal Events" by
  Embrechts, Klüppelberg and Mikosch (Springer, 1997);

"""

import mpmath


__all__ = ['pdf', 'logpdf', 'cdf', 'invcdf', 'sf', 'invsf', 'mean', 'var']


def _validate_params(k, loc, scale):
    if k <= 0:
        raise ValueError('k must be positive')
    if scale <= 0:
        raise ValueError('scale must be positive')
    k = mpmath.mp.mpf(k)
    loc = mpmath.mp.mpf(loc)
    scale = mpmath.mp.mpf(scale)
    return k, loc, scale


def _validate_p(p):
    if p < 0 or p > 1:
        raise ValueError('p must be in the interval [0, 1]')
    return mpmath.mp.mpf(p)


def pdf(x, k, loc, scale):
    """
    Probability density function for the Weibull distribution (for maxima).

    This is a three-parameter version of the distribution.  The more typical
    two-parameter version has just the parameters k and scale.
    """
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        k, loc, scale = _validate_params(k, loc, scale)
        if x == loc:
            if k < 1:
                return mpmath.mp.inf
            elif k == 1:
                return 1/scale
            else:
                return mpmath.mp.zero
        if x > loc:
            return mpmath.mp.zero
        return mpmath.exp(logpdf(x, k, loc, scale))


def logpdf(x, k, loc, scale):
    """
    Log of the PDF of the Weibull distribution (for maxima).

    This is a three-parameter version of the distribution.  The more typical
    two-parameter version has just the parameters k and scale.
    """
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        k, loc, scale = _validate_params(k, loc, scale)
        if x == loc:
            if k < 1:
                return mpmath.mp.inf
            elif k == 1:
                return -mpmath.log(scale)
            else:
                return mpmath.mp.ninf
        if x > loc:
            return mpmath.mp.ninf
        z = (x - loc) / scale
        return (mpmath.log(k) - k*mpmath.log(scale) +
                (k - 1)*mpmath.log(-x+loc) - (-z)**k)


def cdf(x, k, loc, scale):
    """
    Cumulative distribution function for the Weibull distribution (for maxima).

    This is a three-parameter version of the distribution.  The more typical
    two-parameter version has just the parameters k and scale.
    """
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        k, loc, scale = _validate_params(k, loc, scale)
        if x >= loc:
            return mpmath.mp.one
        z = (x - loc) / scale
        return mpmath.exp(-(-z)**k)


def invcdf(p, k, loc, scale):
    """
    Inverse of the CDF for the Weibull distribution (for maxima).

    This is a three-parameter version of the distribution.  The more typical
    two-parameter version has just the parameters k and scale.
    """
    with mpmath.extradps(5):
        p = _validate_p(p)
        k, loc, scale = _validate_params(k, loc, scale)
        z = -mpmath.power(-mpmath.log(p), 1/k)
        x = scale*z + loc
        return x


def sf(x, k, loc, scale):
    """
    Survival function for the Weibull distribution (for maxima).

    This is a three-parameter version of the distribution.  The more typical
    two-parameter version has just the parameters k and scale.
    """
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        k, loc, scale = _validate_params(k, loc, scale)
        if x >= loc:
            return mpmath.mp.zero
        z = (x - loc) / scale
        return -mpmath.expm1(-(-z)**k)


def invsf(p, k, loc, scale):
    """
    Inverse of the survival function for the Weibull distribution (for maxima).

    This is a three-parameter version of the distribution.  The more typical
    two-parameter version has just the parameters k and scale.
    """
    with mpmath.extradps(5):
        p = _validate_p(p)
        k, loc, scale = _validate_params(k, loc, scale)
        z = -mpmath.power(-mpmath.log1p(-p), 1/k)
        x = scale*z + loc
        return x


def mean(k, loc, scale):
    """
    Mean of the Weibull distribution (for maxima).

    This is a three-parameter version of the distribution.  The more typical
    two-parameter version has just the parameters k and scale.
    """
    with mpmath.extradps(5):
        k, loc, scale = _validate_params(k, loc, scale)
        return loc - scale * mpmath.gamma(1 + 1/k)


def var(k, loc, scale):
    """
    Variance of the Weibull distribution (for maxima).

    This is a three-parameter version of the distribution.  The more typical
    two-parameter version has just the parameters k and scale.
    """
    with mpmath.extradps(5):
        k, loc, scale = _validate_params(k, loc, scale)
        v1 = 1 + 1/k
        v2 = 1 + 2/k
        return scale**2 * (mpmath.gamma(v2) - mpmath.gamma(v1)**2)
