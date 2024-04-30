"""
Kumaraswamy probability distribution
------------------------------------

References
----------

Kumaraswamy, P. (1980). "A generalized probability density function for
double-bounded random processes". Journal of Hydrology. 46 (1-2): 79-88.
doi:10.1016/0022-1694(80)90036-0.

Jones, M.C. (2009). "Kumaraswamy's distribution: A beta-type distribution
with some tractability advantages". Statistical Methodology. 6 (1): 70-81.
doi:10.1016/j.stamet.2008.04.001.

"Kumaraswamy distribution". Wikipedia,
https://en.wikipedia.org/wiki/Kumaraswamy_distribution.

"""

from mpmath import mp
from ._common import (_validate_p, _validate_moment_n, _validate_x_bounds,
                      Initial)
from ..fun._powm1 import inv_powm1


__all__ = ['pdf', 'logpdf', 'cdf', 'invcdf', 'sf', 'invsf',
           'mean', 'var', 'median', 'skewness', 'noncentral_moment', 'entropy',
           'nll', 'mle']


def _validate_a_b(a, b):
    if a <= 0 or b <= 0:
        raise ValueError('The shape parameters a and b must be greater '
                         'than 0.')
    return mp.mpf(a), mp.mpf(b)


def pdf(x, a, b):
    """
    Probability density function (PDF) for the Kumaraswamy distribution.
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
        return (a * b * mp.power(x, a - 1)
                * mp.power(-mp.powm1(x, a), b - 1))


def logpdf(x, a, b):
    """
    Natural logarithm of the PDF of the Kumaraswamy distribution.
    """
    with mp.extradps(5):
        a, b = _validate_a_b(a, b)
        x = mp.mpf(x)
        if x < 0 or x > 1:
            return mp.ninf
        return (mp.log(a) + mp.log(b) + (a - 1)*mp.log(x)
                + (b - 1)*mp.log1p(-x**a))


def cdf(x, a, b):
    """
    Cumulative distribution function of the Kumaraswamy distribution.
    """
    with mp.extradps(5):
        a, b = _validate_a_b(a, b)
        x = mp.mpf(x)
        if x < 0:
            return mp.zero
        if x > 1:
            return mp.one
        return -mp.powm1(-mp.powm1(x, a), b)


def sf(x, a, b):
    """
    Survival function of the Kumaraswamy distribution.
    """
    with mp.extradps(5):
        a, b = _validate_a_b(a, b)
        x = mp.mpf(x)
        if x < 0:
            return mp.one
        if x > 1:
            return mp.zero
        return mp.power(-mp.powm1(x, a), b)


def invcdf(p, a, b):
    """
    Inverse of the CDF of the Kumaraswamy distribution.
    """
    with mp.extradps(5):
        a, b = _validate_a_b(a, b)
        p = _validate_p(p)
        if p == 0:
            return mp.zero
        if p == 1:
            return mp.one
        return inv_powm1(-inv_powm1(-p, b), a)


def invsf(p, a, b):
    """
    Inverse of the survival function of the Kumaraswamy distribution.
    """
    with mp.extradps(5):
        a, b = _validate_a_b(a, b)
        p = _validate_p(p)
        if p == 0:
            return mp.one
        if p == 1:
            return mp.zero
        return inv_powm1(-mp.power(p, 1/b), a)


def mean(a, b):
    """
    Mean of the Kumaraswamy distribution.
    """
    with mp.extradps(5):
        a, b = _validate_a_b(a, b)
        return b*mp.beta(1 + 1/a, b)


def var(a, b):
    """
    Variance of the Kumaraswamy distribution.
    """
    with mp.extradps(5):
        a, b = _validate_a_b(a, b)
        return b*mp.beta(1 + 2/a, b) - mean(a, b)**2


def median(a, b):
    """
    Median of the Kumaraswamy distribution.
    """
    with mp.extradps(5):
        a, b = _validate_a_b(a, b)
        return inv_powm1(-mp.power(0.5, 1/b), a)


def noncentral_moment(n, a, b):
    """
    n-th noncentral moment of the Kumaraswamy distribution.

    n must be a nonnegative integer.
    """
    n = _validate_moment_n(n)
    with mp.extradps(5):
        a, b = _validate_a_b(a, b)
        if n == 0:
            return mp.one
        return b * mp.beta(1 + n/a, b)


def skewness(a, b):
    """
    Skewness of the Kumaraswamy distribution.
    """
    with mp.extradps(5):
        a, b = _validate_a_b(a, b)
        m = mean(a, b)
        v = var(a, b)
        mu3p = noncentral_moment(3, a, b)
        return (mu3p - m*(3*v + m**2))/v**1.5


def _harmonic_number(n):
    return mp.digamma(n + 1) + mp.euler


def entropy(a, b):
    """
    Differential entropy of the Kumaraswamy distribution.
    """
    with mp.extradps(5):
        a, b = _validate_a_b(a, b)
        return ((1 - 1/b) + (1 - 1/a)*_harmonic_number(b)
                - mp.log(a) - mp.log(b))


def nll(x, a, b):
    """
    Negative log-likelihood function for the Kumaraswamy distribution.

    `x` must be a sequence of numbers with values in the open interval (0, 1).
    """
    with mp.extradps(5):
        x = _validate_x_bounds(x, low=0, high=1,
                               strict_low=True, strict_high=True)
        a, b = _validate_a_b(a, b)
        return -mp.fsum([logpdf(t, a, b) for t in x])


def _mle_a_eqn(a, x, sumlogx):
    n = len(x)
    s2 = mp.fsum([mp.log1p(-t**a) for t in x])
    s3 = mp.fsum([t**a * mp.log(t) / (-mp.powm1(t, a)) for t in x])
    return n/a + sumlogx + (1 + n/s2)*s3


def _mle_a_eqn_b_fixed(a, b, x, sumlogx):
    n = len(x)
    s3 = mp.fsum([t**a * mp.log(t) / (-mp.powm1(t, a)) for t in x])
    return n/a + sumlogx + (1 - b)*s3


def mle(x, *, a=None, b=None):
    """
    Maximum likelihood estimate for the Kumaraswamy distribution.

    `x` must be a sequence of numbers with values in the open interval (0, 1).

    Returns (a, b), the maximum likelihood estimate for the given data.

    When `a` is not given, the MLE equations are solved numerically,
    and the solver may fail to converge for some inputs.  If this happens,
    a different initial guess for `a` may be given by setting the input
    parameter to `mpsci.distributions.Initial(a0)`, where `a0` is the
    initial guess to use for the estimate of `a`.  The default initial
    guess for `a` is 1.

    """
    with mp.extradps(5):
        x = _validate_x_bounds(x, low=0, high=1,
                               strict_low=True, strict_high=True)
        n = len(x)
        sumlogx = mp.fsum([mp.log(t) for t in x])
        if (a is None or isinstance(a, Initial)) and b is None:
            if isinstance(a, Initial):
                a0 = a.initial
            else:
                a0 = 1
            a = mp.findroot(lambda t: _mle_a_eqn(t, x, sumlogx), a0)
            b = -n / mp.fsum([mp.log1p(-t**a) for t in x])
            return a, b
        if (a is not None and not isinstance(a, Initial)) and b is not None:
            a, b = _validate_a_b(a, b)
            return a, b
        if a is None or isinstance(a, Initial):
            _, b = _validate_a_b(1, b)
            if isinstance(a, Initial):
                a0 = a.initial
            else:
                a0 = 1
            a = mp.findroot(lambda t: _mle_a_eqn_b_fixed(t, b, x, sumlogx), a0)
            return a, b
        # a is fixed, b is not.
        a, _ = _validate_a_b(a, 1)
        b = -n / mp.fsum([mp.log1p(-t**a) for t in x])
        return a, b
