"""
Weibull probability distribution (for maxima)
---------------------------------------------

This is the same distribution as:

* `scipy.stats.weibull_max`;
* the Type III extreme value distribution used in the text "An Introduction
  to Statistical Modeling of Extreme Values" by Stuart Coles (Springer, 2001);
* the Weibull distribution given in the text "Modelling Extremal Events" by
  Embrechts, Klüppelberg and Mikosch (Springer, 1997).

This is the same distributions as `weibull_min`, but with `x` replaced
by `-x`.  The `weibull_min` distribution is the same distribution as:

* `scipy.stats.weibull_min`;
* `numpy.random.Generator.weibull` (restricted to loc=0 and scale=1);
* Wolfram Alpha's `WeibullDistribution`;
* the distribution discussed in the wikipedia article "Weibull distribution"
  (https://en.wikipedia.org/wiki/Weibull_distribution);
* the Weibull distribution in the text "Statistical Distribution" (fourth ed.)
  by Forbes, Evans, Hastings and Peacock (Wiley, 2011).

"""

from mpmath import mp
from ..stats import pmean
from ._common import (_validate_p, _validate_moment_n, _validate_x_bounds,
                      _median)
from ._weibull_common import _validate_params, _mle_k_eqn1, _mle_k_eqn2


__all__ = ['pdf', 'logpdf', 'cdf', 'invcdf', 'sf', 'invsf',
           'mode', 'mean', 'var', 'skewness', 'kurtosis', 'entropy',
           'noncentral_moment', 'nll', 'mle']


def pdf(x, k, loc, scale):
    """
    Probability density function for the Weibull distribution (for maxima).

    This is a three-parameter version of the distribution.  The more typical
    two-parameter version has just the parameters k and scale.
    """
    with mp.extradps(5):
        x = mp.mpf(x)
        k, loc, scale = _validate_params(k, loc, scale)
        if x == loc:
            if k < 1:
                return mp.inf
            elif k == 1:
                return 1/scale
            else:
                return mp.zero
        if x > loc:
            return mp.zero
        return mp.exp(logpdf(x, k, loc, scale))


def logpdf(x, k, loc, scale):
    """
    Log of the PDF of the Weibull distribution (for maxima).

    This is a three-parameter version of the distribution.  The more typical
    two-parameter version has just the parameters k and scale.
    """
    with mp.extradps(5):
        x = mp.mpf(x)
        k, loc, scale = _validate_params(k, loc, scale)
        if x == loc:
            if k < 1:
                return mp.inf
            elif k == 1:
                return -mp.log(scale)
            else:
                return mp.ninf
        if x > loc:
            return mp.ninf
        z = (x - loc) / scale
        return (mp.log(k) - k*mp.log(scale) +
                (k - 1)*mp.log(-x+loc) - (-z)**k)


def cdf(x, k, loc, scale):
    """
    Cumulative distribution function for the Weibull distribution (for maxima).

    This is a three-parameter version of the distribution.  The more typical
    two-parameter version has just the parameters k and scale.
    """
    with mp.extradps(5):
        x = mp.mpf(x)
        k, loc, scale = _validate_params(k, loc, scale)
        if x >= loc:
            return mp.one
        z = (x - loc) / scale
        return mp.exp(-(-z)**k)


def invcdf(p, k, loc, scale):
    """
    Inverse of the CDF for the Weibull distribution (for maxima).

    This is a three-parameter version of the distribution.  The more typical
    two-parameter version has just the parameters k and scale.
    """
    with mp.extradps(5):
        p = _validate_p(p)
        k, loc, scale = _validate_params(k, loc, scale)
        z = -mp.power(-mp.log(p), 1/k)
        x = scale*z + loc
        return x


def sf(x, k, loc, scale):
    """
    Survival function for the Weibull distribution (for maxima).

    This is a three-parameter version of the distribution.  The more typical
    two-parameter version has just the parameters k and scale.
    """
    with mp.extradps(5):
        x = mp.mpf(x)
        k, loc, scale = _validate_params(k, loc, scale)
        if x >= loc:
            return mp.zero
        z = (x - loc) / scale
        return -mp.expm1(-(-z)**k)


def invsf(p, k, loc, scale):
    """
    Inverse of the survival function for the Weibull distribution (for maxima).

    This is a three-parameter version of the distribution.  The more typical
    two-parameter version has just the parameters k and scale.
    """
    with mp.extradps(5):
        p = _validate_p(p)
        k, loc, scale = _validate_params(k, loc, scale)
        z = -mp.power(-mp.log1p(-p), 1/k)
        x = scale*z + loc
        return x


def mode(k, loc, scale):
    """
    Mode of the Weibull distribution (for maxima).

    This is a three-parameter version of the distribution.  The more typical
    two-parameter version has just the parameters k and scale.
    """
    with mp.extradps(5):
        k, loc, scale = _validate_params(k, loc, scale)
        m = scale * mp.power((k - 1)/k, 1/k) if k > 1 else 0
        return loc - m


def mean(k, loc, scale):
    """
    Mean of the Weibull distribution (for maxima).

    This is a three-parameter version of the distribution.  The more typical
    two-parameter version has just the parameters k and scale.
    """
    with mp.extradps(5):
        k, loc, scale = _validate_params(k, loc, scale)
        return loc - scale * mp.gamma(1 + 1/k)


def var(k, loc, scale):
    """
    Variance of the Weibull distribution (for maxima).

    This is a three-parameter version of the distribution.  The more typical
    two-parameter version has just the parameters k and scale.
    """
    with mp.extradps(5):
        k, loc, scale = _validate_params(k, loc, scale)
        v1 = 1 + 1/k
        v2 = 1 + 2/k
        return scale**2 * (mp.gamma(v2) - mp.gamma(v1)**2)


def skewness(k, loc, scale):
    """
    Skewness of the Weibull distribution (for maxima).

    This is a three-parameter version of the distribution.  The more typical
    two-parameter version has just the parameters k and scale.
    """
    with mp.extradps(5):
        k, loc, scale = _validate_params(k, loc, scale)
        g1 = mp.gamma(1 + 1/k)
        g2 = mp.gamma(1 + 2/k)
        g3 = mp.gamma(1 + 3/k)
        return -(g3 - 3*g1*g2 + 2*g1**3) / mp.power(g2 - g1**2, 1.5)


def kurtosis(k, loc, scale):
    """
    Excess kurtosis of the Weibull distribution (for minima).

    This is a three-parameter version of the distribution.  The more typical
    two-parameter version has just the parameters k and scale.
    """
    with mp.extradps(5):
        k, loc, scale = _validate_params(k, loc, scale)
        g1 = mp.gamma(1 + 1/k)
        g2 = mp.gamma(1 + 2/k)
        g3 = mp.gamma(1 + 3/k)
        g4 = mp.gamma(1 + 4/k)
        den = (g2 - g1**2)**2
        return (-6*g1**4 + 12*g1**2*g2 - 3*g2**2 - 4*g1*g3 + g4) / den


def entropy(k, loc, scale):
    """
    Differential entropy of the Weibull distribution (for maxima).

    This is a three-parameter version of the distribution.  The more typical
    two-parameter version has just the parameters k and scale.
    """
    with mp.extradps(5):
        k, loc, scale = _validate_params(k, loc, scale)
        return mp.euler*(1 - 1/k) + mp.log(scale) - mp.log(k) + 1


def _standard_noncentral_moment(n, k):
    with mp.extradps(5):
        if n == 0:
            return mp.one
        return (-1)**(n % 2) * mp.gamma(1 + mp.mpf(n)/k)


def noncentral_moment(n, k, loc=0, scale=1):
    """
    Noncentral moment of the Weibull distribution (for maxima).

    The value is also known as the raw moment.
    """
    with mp.extradps(5):
        _validate_moment_n(n)
        k, loc, sigma = _validate_params(k, loc, scale)
        if n == 0:
            return mp.one
        terms = [(mp.binomial(n, i) * mp.power(loc, n - i) * mp.power(scale, i)
                  * _standard_noncentral_moment(i, k))
                 for i in range(n + 1)]
        return mp.fsum(terms)


def nll(x, k, loc, scale):
    """
    Negative log-likelihood function for the Weibull(min) distribution.
    """
    _validate_params(k, loc, scale)
    with mp.extradps(5):
        x = _validate_x_bounds(x, high=loc, strict_high=True, highname='loc')
        return -mp.fsum([logpdf(t, k, loc, scale) for t in x])


def mle(x, k=None, loc=None, scale=None):
    """
    Maximum likelihood estimate of the Weibull(max) distribution parameters.

    `loc` must be given.

    Return value is (k, loc, scale).
    """
    if loc is None:
        raise ValueError("The 'loc' parameter must be given explicitly.")
    with mp.extradps(5):
        x = _validate_x_bounds(x, high=loc, strict_high=True, highname='loc')
        # Transform x to loc - x.  After that transformation, this
        # function is the same as weibull_min. (XXX FIXME: There is some
        # D.R.Y. clean up to be done here.)
        loc = mp.mpmathify(loc)
        x = [-t + loc for t in x]

        if k is not None and k <= 0:
            raise ValueError('k must be greater than 0')
        if scale is not None and scale <= 0:
            raise ValueError('scale must be greater than 0')

        if k is None and scale is None:
            # Solve for k and scale.
            # TO DO: Is there a better guess for k than 1?
            k_hat = mp.findroot(lambda k: _mle_k_eqn1(k, x), 1.0)
            scale_hat = pmean(x, k_hat)
        elif k is not None and scale is None:
            # Solve for scale.
            k_hat = mp.mpf(k)
            scale_hat = pmean(x, k_hat)
        elif k is None and scale is not None:
            # Solve for k only.
            scale_hat = mp.mpf(scale)
            # Use the formula for the median,
            #    median = scale * ln(2)**(1/k)
            # to derive the initial guess for k.
            med = _median(x)
            lnr = mp.log(med/scale_hat)
            if lnr >= 0:
                k0 = 1.0
            else:
                k0 = mp.log(mp.log(2))/lnr
            k_hat = mp.findroot(lambda k: _mle_k_eqn2(k, x, scale_hat), k0)
        else:
            # Both k and scale are not None--nothing to do.
            k_hat = mp.mpf(k)
            scale_hat = mp.mpf(scale)

    return k_hat, loc, scale_hat
