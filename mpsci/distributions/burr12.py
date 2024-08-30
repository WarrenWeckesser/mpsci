"""
Burr type XII probability distribution
--------------------------------------

"""

from mpmath import mp
from mpsci.stats import mean as _fmean
from mpsci.distributions._common import (_validate_p,  _validate_x_bounds,
                                         Initial, isfixed)


__all__ = ['support', 'pdf', 'logpdf',
           'cdf', 'invcdf', 'logcdf', 'sf', 'invsf', 'logsf',
           'mean', 'var', 'median', 'mode',
           'nll']


def _validate_params(c, d, scale):
    if c <= 0:
        raise ValueError('c must be greater than 0.')
    if d <= 0:
        raise ValueError('d must be greater than 0.')
    if scale <= 0:
        raise ValueError('scale must be greater than 0.')
    return mp.mpf(c), mp.mpf(d), mp.mpf(scale)


def pdf(x, c, d, scale):
    """
    Probability density function of the Burr type XII distribution.
    """
    with mp.extradps(5):
        c, d, scale = _validate_params(c, d, scale)
        x = mp.mpf(x)
        if x <= 0:
            return mp.zero
        z = x/scale
        return c*d*z**(c - 1)/scale / (1 + z**c)**(d+1)


def logpdf(x, c, d, scale):
    """
    Natural logarithm of the PDF of the Burr type XII distribution.
    """
    with mp.extradps(5):
        c, d, scale = _validate_params(c, d, scale)
        x = mp.mpf(x)
        if x <= 0:
            return mp.ninf
        return (mp.log(c) + mp.log(d) + (c - 1)*mp.log(x)
                - c*mp.log(scale) - (d + 1)*mp.log1p((x / scale)**c))


def cdf(x, c, d, scale):
    """
    Cumulative distribution function of the Burr type XII distribution.
    """
    with mp.extradps(5):
        c, d, scale = _validate_params(c, d, scale)
        x = mp.mpf(x)
        if x <= 0:
            return mp.zero
        # TO DO: See if the use of logsf (as in scipy) is worthwhile.
        return 1 - sf(x, c, d, scale)


def invcdf(p, c, d, scale):
    """
    Inverse of the CDF of the Burr type XII distribution.
    """
    with mp.extradps(5):
        c, d, scale = _validate_params(c, d, scale)
        p = _validate_p(p)
        return scale * mp.powm1(1 - p, -1/d)**(1/c)


def logcdf(x, c, d, scale):
    """
    Natural logarithm of the CDF of the Burr type XII distribution.
    """
    with mp.extradps(5):
        c, d, scale = _validate_params(c, d, scale)
        x = mp.mpf(x)
        if x <= 0:
            return mp.ninf
        return mp.log1p(-sf(x, c, d, scale))


def sf(x, c, d, scale):
    """
    Survival function of the Burr type XII distribution.
    """
    with mp.extradps(5):
        c, d, scale = _validate_params(c, d, scale)
        x = mp.mpf(x)
        if x <= 0:
            return mp.one
        return (1 + (x/scale)**c)**(-d)


def invsf(p, c, d, scale):
    """
    Inverse of the survival function of the Burr type XII distribution.
    """
    with mp.extradps(5):
        c, d, scale = _validate_params(c, d, scale)
        p = _validate_p(p)
        return scale * mp.powm1(p, -1/d)**(1/c)


def logsf(x, c, d, scale):
    """
    Natural logarithm of the survival function of the Burr type XII distribution.
    """
    with mp.extradps(5):
        c, d, scale = _validate_params(c, d, scale)
        x = mp.mpf(x)
        if x <= 0:
            return mp.zero
        return -d*mp.log1p((x/scale)**c)


def support(c, d, scale):
    """
    Support of the Burr type XII distribution.
    """
    with mp.extradps(5):
        c, d, scale = _validate_params(c, d, scale)
        return (mp.zero, mp.inf)


def mean(c, d, scale):
    """
    Mean of the Burr type XII distribution.
    """
    with mp.extradps(5):
        c, d, scale = _validate_params(c, d, scale)
        if c*d <= 1:
            return mp.nan
        return d*mp.beta(d - 1/c, 1 + 1/c)*scale


def var(c, d, scale):
    """
    Variance of the Burr type XII distribution.
    """
    with mp.extradps(5):
        c, d, scale = _validate_params(c, d, scale)
        if c*d <= 2:
            return mp.nan
        mu1 = mean(c, d, 1)
        mu2 = d*mp.beta(d - 2/c, 1 + 2/c)
        return scale**2 * (-mu1**2 + mu2)


def median(c, d, scale):
    """
    Median of the Burr type XII distribution.
    """
    with mp.extradps(5):
        c, d, scale = _validate_params(c, d, scale)
        return scale * (2**(1/d) - 1)**(1/c)


def mode(c, d, scale):
    """
    Mode of the Burr type XII distribution.
    """
    with mp.extradps(5):
        c, d, scale = _validate_params(c, d, scale)
        if c <= 1:
            return mp.zero
        return scale*((c - 1)/(d*c + 1))**(1/c)


def nll(x, c, d, scale):
    """
    Negative log-likelihood for the Burr type XII distribution.

    `x` must be a sequence of numbers.
    """
    with mp.extradps(5):
        c, d, scale = _validate_params(c, d, scale)
        x = _validate_x_bounds(x, low=0, strict_low=True, high=mp.inf)
        return -mp.fsum([logpdf(xi, c, d, scale) for xi in x])


# Work in progress...


def _nll_grad(x, c, d, scale):
    n = len(x)
    sumlogx = mp.fsum([mp.log(xi) for xi in x])
    z = [(xi/scale) for xi in x]
    r = [zi**c/(1 + zi**c) for zi in z]
    return [n/c - n*mp.log(scale) + sumlogx
            - (d + 1)*mp.fsum([ri*mp.log(zi) for ri, zi in zip(r, z)]),
            n/d - mp.fsum([mp.log1p(zi**c) for zi in z]),
            (-c*n + c*(d + 1)*mp.fsum(r))/scale]


def _mle_c_d_scale(x, c0=1, d0=1, scale0=1):

    def _first_order_eqs(c, d, scale):
        return _nll_grad(x, c, d, scale)

    c, d, scale = mp.findroot(_first_order_eqs, [c0, d0, scale0])
    return c, d, scale


def _mle_d_scale(x, c, d0=1, scale0=1):

    def _first_order_eqs(d, scale):
        return _nll_grad(x, c, d, scale)[1:]

    d, scale = mp.findroot(_first_order_eqs, [d0, scale0])
    return d, scale


def _mle_c_scale(x, d, c0=1, scale0=1):

    def _first_order_eqs(c, scale):
        v = _nll_grad(x, c, d, scale)
        return [v[0], v[2]]

    c, scale = mp.findroot(_first_order_eqs, [c0, scale0])
    return c, scale


@mp.extradps(5)
def _scale_eq_c_and_d_fixed(scale, x, c, d):
    scale = mp.mpf(scale)
    x = [mp.mpf(t) for t in x]
    c = mp.mpf(c)
    d = mp.mpf(d)
    q = [(t/scale)**c for t in x]
    w = [t/(1 + t) for t in q]
    m = _fmean(w)
    return 1 - (d + 1)*m


@mp.extradps(5)
def _find_scale_bracket_c_and_d_fixed(x, c, d):
    scale0 = mp.zero
    scale1 = mp.eps
    v = _scale_eq_c_and_d_fixed(scale1, x, c, d)
    while v < 0:
        scale0 = scale1
        scale1 *= 4
        v = _scale_eq_c_and_d_fixed(scale1, x, c, d)
    return scale0, scale1


@mp.extradps(5)
def _c_eq_d_and_scale_fixed(c, x, d, scale):
    x = [mp.mpf(t) for t in x]
    c = mp.mpf(c)
    d = mp.mpf(d)
    scale = mp.mpf(scale)
    n = len(x)
    z = [xi/scale for xi in x]
    logz = [mp.log(zi) for zi in z]
    return (n/c + mp.fsum(logz)
            - (d + 1)*mp.fsum([mp.sigmoid(c*mp.log(zi))*mp.log(zi) for zi in z]))


@mp.extradps(5)
def _find_c_bracket_d_and_scale_fixed(x, d, scale):
    c0 = mp.zero
    c1 = mp.eps
    v = _c_eq_d_and_scale_fixed(c1, x, d, scale)
    while v > 0:
        c0 = c1
        c1 *= 4
        v = _c_eq_d_and_scale_fixed(c1, x, d, scale)
    return c0, c1


@mp.extradps(5)
def _c_eq_scale_fixed(c, x, scale):
    x = [mp.mpf(t) for t in x]
    c = mp.mpf(c)
    scale = mp.mpf(scale)
    n = len(x)
    z = [xi/scale for xi in x]
    logz = [mp.log(zi) for zi in z]
    d = 1/_fmean([mp.log1p(zi**c) for zi in z])
    return (n/c + mp.fsum(logz)
            - (d + 1)*mp.fsum([mp.sigmoid(c*mp.log(zi))*mp.log(zi) for zi in z]))


@mp.extradps(5)
def _find_c_bracket_scale_fixed(x, scale):
    c0 = mp.zero
    c1 = mp.eps
    v = _c_eq_scale_fixed(c1, x, scale)
    while v > 0:
        c0 = c1
        c1 *= 4
        v = _c_eq_scale_fixed(c1, x, scale)
    return c0, c1


@mp.extradps(5)
def mle(x, *, c=None, d=None, scale=None):
    """
    Maximum likelihood estimate for the Burr type XII distribution.

    `x` must be a sequence of numbers.
    """
    # The location parameter isn't implemented, so the support
    # is (0, inf).
    x = _validate_x_bounds(x, low=0, strict_low=True, high=mp.inf)
    c_fixed = isfixed(c)
    d_fixed = isfixed(d)
    scale_fixed = isfixed(scale)

    if c_fixed and d_fixed and scale_fixed:
        # All parameters fixed, nothing to do.
        c, d, scale = _validate_params(c, d, scale)
        return c, d, scale

    elif c_fixed and not d_fixed and not scale_fixed:
        # Not working???
        d0 = d.initial if isinstance(d, Initial) else 1
        scale0 = scale.initial if isinstance(scale, Initial) else 1
        c, d0, scale0 = _validate_params(c, d0, scale0)
        d, scale = _mle_d_scale(x, c, d0, scale0)
        return c, d, scale

    elif c_fixed and d_fixed and not scale_fixed:
        # Fit the scale; c and d are fixed.
        scale0 = scale.initial if isinstance(scale, Initial) else 1
        c, d, scale0 = _validate_params(c, d, scale0)
        scale_low, scale_high = _find_scale_bracket_c_and_d_fixed(x, c, d)
        scale = mp.findroot(lambda s: _scale_eq_c_and_d_fixed(s, x, c, d),
                            (scale_low, scale_high), method='anderson')
        return c, d, scale

    elif c_fixed and not d_fixed and scale_fixed:
        d0 = d.initial if isinstance(d, Initial) else 1
        # d0 is validated, but it is ignored, because we have an explicit
        # formula in this case and there is no need for an iterative solver.
        c, d0, scale = _validate_params(c, d0, scale)
        d = 1/_fmean([mp.log1p((xi/scale)**c) for xi in x])
        return c, d, scale

    elif not c_fixed and d_fixed and scale_fixed:
        c0 = c.initial if isinstance(c, Initial) else 1
        c0, d, scale = _validate_params(c0, d, scale)
        c_low, c_high = _find_c_bracket_d_and_scale_fixed(x, d, scale)
        c = mp.findroot(lambda t: _c_eq_d_and_scale_fixed(t, x, d, scale),
                        (c_low, c_high), method='anderson')
        return c, d, scale

    elif not c_fixed and not d_fixed and scale_fixed:
        # Only the scale is fixed.
        c0 = c.initial if isinstance(c, Initial) else 1
        d0 = d.initial if isinstance(d, Initial) else 1
        c0, d0, scale = _validate_params(c0, d0, scale)
        c_low, c_high = _find_c_bracket_scale_fixed(x, scale)
        c = mp.findroot(lambda t: _c_eq_scale_fixed(t, x, scale),
                        (c_low, c_high), method='anderson')
        d = 1/_fmean([mp.log1p((xi/scale)**c) for xi in x])
        return c, d, scale

    elif not c_fixed and d_fixed and not scale_fixed:
        # work in progress...
        c0 = c.initial if isinstance(c, Initial) else 1
        scale0 = scale.initial if isinstance(scale, Initial) else 1
        c0, d, scale0 = _validate_params(c0, d, scale0)
        c, scale = _mle_c_scale(x, d, c0, scale0)
        return c, d, scale

    elif not c_fixed and not d_fixed and not scale_fixed:
        # Fit c, d and scale
        c0 = c.initial if isinstance(c, Initial) else 1
        d0 = d.initial if isinstance(d, Initial) else 1
        scale0 = scale.initial if isinstance(scale, Initial) else 1
        c0, d0, scale0 = _validate_params(c0, d0, scale0)
        c, d, scale = _mle_c_d_scale(x, c0, d0, scale0)
        return c, d, scale

    else:
        raise NotImplementedError
