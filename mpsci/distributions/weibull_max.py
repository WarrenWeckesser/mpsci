"""
Weibull probability distribution (for maxima)
---------------------------------------------

This is the same distribution as:

* `scipy.stats.weibull_max`;
* the Type III extreme value distribution used in the text "An Introduction
  to Statistical Modeling of Extreme Values" by Stuart Coles (Springer, 2001);
* the Weibull distribution given in the text "Modelling Extremal Events" by
  Embrechts, Klüppelberg and Mikosch (Springer, 1997).

"""

import mpmath
from ..stats import pmean
from ._common import _validate_p, _median
from ._weibull_common import _validate_params, _mle_k_eqn1, _mle_k_eqn2


__all__ = ['pdf', 'logpdf', 'cdf', 'invcdf', 'sf', 'invsf',
           'mean', 'var', 'skewness',
           'nll', 'mle']


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


def skewness(k, loc, scale):
    """
    Skewness of the Weibull distribution (for maxima).

    This is a three-parameter version of the distribution.  The more typical
    two-parameter version has just the parameters k and scale.
    """
    with mpmath.extradps(5):
        k, loc, scale = _validate_params(k, loc, scale)
        g1 = mpmath.gamma(1 + 1/k)
        g2 = mpmath.gamma(1 + 2/k)
        g3 = mpmath.gamma(1 + 3/k)
        return -(g3 - 3*g1*g2 + 2*g1**3) / mpmath.power(g2 - g1**2, 1.5)


def kurtosis(k, loc, scale):
    """
    Excess kurtosis of the Weibull distribution (for minima).

    This is a three-parameter version of the distribution.  The more typical
    two-parameter version has just the parameters k and scale.
    """
    with mpmath.extradps(5):
        k, loc, scale = _validate_params(k, loc, scale)
        g1 = mpmath.gamma(1 + 1/k)
        g2 = mpmath.gamma(1 + 2/k)
        g3 = mpmath.gamma(1 + 3/k)
        g4 = mpmath.gamma(1 + 4/k)
        den = (g2 - g1**2)**2
        return (-6*g1**4 + 12*g1**2*g2 - 3*g2**2 - 4*g1*g3 + g4) / den


def _validate_x(x, loc=0):
    if any(t >= loc for t in x):
        raise ValueError(f'All values in x must be less than loc ({loc}).')


def nll(x, k, loc, scale):
    """
    Negative log-likelihood function for the Weibull(min) distribution.
    """
    _validate_params(k, loc, scale)
    _validate_x(x, loc=loc)
    with mpmath.extradps(5):
        return -mpmath.fsum([logpdf(t, k, loc, scale) for t in x])


def mle(x, k=None, loc=None, scale=None):
    """
    Maximum likelihood estimate of the Weibull(min) distribution parameters.

    `loc` must be given.

    Return value is (k, loc, scale).
    """
    if loc is None:
        raise ValueError("The 'loc' parameter must be given explicitly.")
    _validate_x(x, loc)
    with mpmath.extradps(5):
        # Transform x to loc - x.  After that transformation, this
        # function is the same as weibull_min. (XXX FIXME: There is some
        # D.R.Y. clean up to be done here.)
        loc = mpmath.mpf(loc)
        x = [-mpmath.mpf(1.0*t) + loc for t in x]

        if k is not None and k <= 0:
            raise ValueError('k must be greater than 0')
        if scale is not None and scale <= 0:
            raise ValueError('scale must be greater than 0')

        if k is None and scale is None:
            # Solve for k and scale.
            # TO DO: Is there a better guess for k than 1?
            k_hat = mpmath.findroot(lambda k: _mle_k_eqn1(k, x), 1.0)
            scale_hat = pmean(x, k_hat)
        elif k is not None and scale is None:
            # Solve for scale.
            k_hat = mpmath.mpf(k)
            scale_hat = pmean(x, k_hat)
        elif k is None and scale is not None:
            # Solve for k only.
            scale_hat = mpmath.mpf(scale)
            # Use the formula for the median,
            #    median = scale * ln(2)**(1/k)
            # to derive the initial guess for k.
            med = _median(x)
            lnr = mpmath.log(med/scale_hat)
            if lnr >= 0:
                k0 = 1.0
            else:
                k0 = mpmath.log(mpmath.log(2))/lnr
            k_hat = mpmath.findroot(lambda k: _mle_k_eqn2(k, x, scale_hat), k0)
        else:
            # Both k and scale are not None--nothing to do.
            k_hat = mpmath.mpf(k)
            scale_hat = mpmath.mpf(scale)

    return k_hat, loc, scale_hat
