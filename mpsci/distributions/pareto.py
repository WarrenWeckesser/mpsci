"""
Pareto probability distribution (type I)
----------------------------------------

This module implements functions for the Pareto distribution.
The wikipedia article [1]_ refers to this as the *Type I*
Pareto distribution, but this implementation includes a location
parameter in addition to shape and scale parameters shown in the
wikipedia page.

.. [1] Pareto distribution, Wikipedia,
       https://en.wikipedia.org/wiki/Pareto_distribution

"""
from mpmath import mp
from ..fun import inv_powm1
from ._common import _validate_p, _validate_x_bounds, Initial


__all__ = ['pdf', 'logpdf', 'cdf', 'invcdf', 'sf', 'invsf',
           'mean', 'var', 'entropy', 'nll', 'mle']


def _validate_params(b, loc, scale):
    if b <= 0:
        raise ValueError('b must be positive')
    if scale <= 0:
        raise ValueError('scale must be positive')
    return mp.mpf(b), mp.mpf(loc), mp.mpf(scale)


def pdf(x, b, loc=0, scale=1):
    """
    Probability density function for the Pareto distribution (type I).

    """
    b, loc, scale = _validate_params(b, loc, scale)
    with mp.extradps(5):
        x = mp.mpf(x)
        lb = loc + scale
        if x < lb:
            return mp.zero
        z = (x - loc)/scale
        return b*z**(-b - 1)/scale


def logpdf(x, b, loc=0, scale=1):
    """
    Logarithm of the PDF for the Pareto distribution (type I).

    """
    b, loc, scale = _validate_params(b, loc, scale)
    with mp.extradps(5):
        x = mp.mpf(x)
        lb = loc + scale
        if x < lb:
            return mp.ninf
        # z = (x - loc)/scale
        return mp.log(b) - (b + 1)*mp.log(x - loc) + b*mp.log(scale)


def cdf(x, b, loc=0, scale=1):
    """
    Cumulative distribution function for the Pareto distribution (type I).

    """
    with mp.extradps(5):
        b, loc, scale = _validate_params(b, loc, scale)
        x = mp.mpf(x)
        lb = loc + scale
        if x < lb:
            return mp.zero
        z = (x - loc)/scale
        return -mp.powm1(z, -b)


def invcdf(p, b, loc=0, scale=1):
    """
    Inverse of the CDF of the Pareto distribution (type I).

    """
    with mp.extradps(5):
        b, loc, scale = _validate_params(b, loc, scale)
        p = _validate_p(p)
        return loc + scale*inv_powm1(-p, -b)


def sf(x, b, loc=0, scale=1):
    """
    Survival function for the Pareto distribution (type I).

    """
    with mp.extradps(5):
        b, loc, scale = _validate_params(b, loc, scale)
        x = mp.mpf(x)
        lb = loc + scale
        if x < lb:
            return mp.one
        z = (x - loc)/scale
        return mp.power(z, -b)


def invsf(p, b, loc=0, scale=1):
    """
    Inverse of the survival function of the Pareto distribution (type I).

    """
    with mp.extradps(5):
        b, loc, scale = _validate_params(b, loc, scale)
        p = _validate_p(p)
        return loc + scale*mp.power(p, -1/b)


def mean(b, loc=0, scale=1):
    """
    Mean of the Pareto distribution (type I).

    """
    with mp.extradps(5):
        b, loc, scale = _validate_params(b, loc, scale)
        if b <= 1:
            return mp.inf
        return loc + scale*b/(b - 1)


def var(b, *, loc=0, scale=1):
    """
    Variance of the Pareto distribution (type I).

    """
    with mp.extradps(5):
        b, loc, scale = _validate_params(b, loc, scale)
        if b <= 2:
            return mp.nan
        return scale**2*b/(b - 1)**2/(b - 2)


def entropy(b, loc=0, scale=1):
    """
    Differential entropy of the Pareto distribution (type I).
    """
    with mp.extradps(5):
        b, loc, scale = _validate_params(b, loc, scale)
        return 1 + 1/b + mp.log(scale/b)


def nll(x, b, loc=0, scale=1):
    """
    Negative log-likelihood function for the Pareto distribution (type I).

    `x` must be a sequence of numbers, and each value in `x` must greater
    than or equal to `loc` + `scale`.
    """
    with mp.extradps(5):
        b, loc, scale = _validate_params(b, loc, scale)
        x = _validate_x_bounds(x, low=loc+scale, strict_low=False,
                               lowname='loc+scale')
        return -mp.fsum(logpdf(t, b, loc=loc, scale=scale) for t in x)


def _is_fixed(obj):
    return obj is not None and not isinstance(obj, Initial)


def mle(x, *, b=None, loc=None, scale=None):
    """
    Maximum likelihood estimation for the Pareto distribution (type I).

    `x` must be a sequence of numbers.

    Currently the function does not implement a good heuristic for
    the starting point in the numerical solver.  The default (1 for
    each parameter that is not fixed) is typically a bad guess and will
    likely result in the numerical solver failing to converge to a
    solution.
    """
    b_fixed = _is_fixed(b)
    loc_fixed = _is_fixed(loc)
    scale_fixed = _is_fixed(scale)
    with mp.extradps(5):
        n = len(x)

        if b_fixed and loc_fixed and scale_fixed:
            # All parameters are fixed.
            b_hat, loc_hat, scale_hat = _validate_params(b, loc, scale)
            x = _validate_x_bounds(x, low=loc_hat + scale_hat, high=mp.inf)
            return b_hat, loc_hat, scale_hat

        if not b_fixed:
            if not loc_fixed and not scale_fixed:
                # All parameters are free.
                x = _validate_x_bounds(x, low=mp.ninf, high=mp.inf)
                x1 = min(x)

                def mle_eqns(b, scale):
                    s1 = mp.fsum([mp.log(t - x1 + scale) for t in x])
                    eq1 = n/b + n*mp.log(scale) - s1
                    s2 = mp.fsum([1/(t - x1 + scale) for t in x])
                    eq2 = n*b/scale - (b + 1)*s2
                    return eq1, eq2

                b0 = b.initial if isinstance(b, Initial) else 1
                scale0 = scale.initial if isinstance(scale, Initial) else 1
                b_hat, scale_hat = mp.findroot(mle_eqns, [b0, scale0])
                loc_hat = x1 - scale_hat
                return b_hat, loc_hat, scale_hat

            if not scale_fixed:
                # b and scale are free, loc is fixed.
                loc = mp.mpf(loc)
                x = _validate_x_bounds(x, low=loc, high=mp.inf)
                x1 = min(x)
                scale_hat = x1 - loc
                s1 = mp.fsum([mp.log(t - loc) for t in x])
                b_hat = 1/(s1/n - mp.log(scale_hat))
                return b_hat, loc, scale_hat

            # b and loc are free, scale is fixed.
            _, _, scale = _validate_params(1, 1, scale)
            x = _validate_x_bounds(x, low=mp.ninf, high=mp.inf)
            x1 = min(x)
            loc_hat = x1 - scale
            s1 = mp.fsum([mp.log(t - loc_hat) for t in x])
            b_hat = 1/(s1/n - mp.log(scale))
            return b_hat, loc_hat, scale

        # b is fixed
        if not loc_fixed and not scale_fixed:
            # b is fixed, loc and scale are free.
            b, _, _ = _validate_params(b, 0, 1)
            x = _validate_x_bounds(x, low=mp.ninf, high=mp.inf)
            x1 = min(x)

            def mle_eqn(scale):
                s = mp.fsum([1/(t - x1 + scale) for t in x])
                return n*b/scale - (b + 1)*s

            scale0 = scale.initial if isinstance(scale, Initial) else 1
            scale_hat = mp.findroot(mle_eqn, scale0)
            loc_hat = x1 - scale_hat
            return b, loc_hat, scale_hat

        if loc_fixed:
            # b and loc are fixed, scale is free.
            b, loc, _ = _validate_params(b, loc, 1)
            x = _validate_x_bounds(x, low=loc, high=mp.inf)
            x1 = min(x)
            scale_hat = x1 - loc
            return b, loc, scale_hat

        # b and scale are fixed, loc is free.
        b, _, scale = _validate_params(b, 0, scale)
        x1 = min([mp.mpf(t) for t in x])
        loc_hat = x1 - scale
        return b, loc_hat, scale
