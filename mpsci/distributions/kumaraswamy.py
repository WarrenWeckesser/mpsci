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
import mpmath
from ._common import _validate_p
from ..fun._powm1 import inv_powm1


__all__ = ['pdf', 'logpdf', 'cdf', 'invcdf', 'sf', 'invsf',
           'mean', 'var', 'median']


def _validate_a_b(a, b):
    if a <= 0 or b <= 0:
        raise ValueError('The shape parameters a and b must be greater '
                         'than 0.')


def pdf(x, a, b):
    """
    Probability density function (PDF) for the Kumaraswamy distribution.
    """
    _validate_a_b(a, b)
    if x < 0 or x > 1:
        return mpmath.mp.zero
    if x == 0 and a < 1:
        return mpmath.inf
    if x == 1 and b < 1:
        return mpmath.inf
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        a = mpmath.mpf(a)
        b = mpmath.mpf(b)
        return (a * b * mpmath.power(x, a - 1)
                * mpmath.power(-mpmath.powm1(x, a), b - 1))


def logpdf(x, a, b):
    """
    Natural logarithm of the PDF of the Kumaraswamy distribution.
    """
    _validate_a_b(a, b)
    x = mpmath.mpf(x)
    a = mpmath.mpf(a)
    b = mpmath.mpf(b)
    if x < 0 or x > 1:
        return -mpmath.mp.inf
    with mpmath.extradps(5):
        return (mpmath.log(a) + mpmath.log(b) + (a - 1)*mpmath.log(x)
                + (b - 1)*mpmath.log(-mpmath.powm1(x, a)))


def cdf(x, a, b):
    """
    Cumulative distribution function of the beta distribution.
    """
    _validate_a_b(a, b)
    if x < 0:
        return mpmath.mp.zero
    if x > 1:
        return mpmath.mp.one
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        a = mpmath.mpf(a)
        b = mpmath.mpf(b)
        return -mpmath.powm1(-mpmath.powm1(x, a), b)


def sf(x, a, b):
    """
    Survival function of the beta distribution.
    """
    _validate_a_b(a, b)
    if x < 0:
        return mpmath.mp.one
    if x > 1:
        return mpmath.mp.zero
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        a = mpmath.mpf(a)
        b = mpmath.mpf(b)
        return mpmath.power(-mpmath.powm1(x, a), b)


def invcdf(p, a, b):
    """
    Inverse of the CDF of the beta distribution.
    """
    _validate_a_b(a, b)
    with mpmath.extradps(5):
        p = _validate_p(p)
        if p == 0:
            return mpmath.mp.zero
        if p == 1:
            return mpmath.mp.one
        a = mpmath.mpf(a)
        b = mpmath.mpf(b)
        return inv_powm1(-inv_powm1(-p, b), a)


def invsf(p, a, b):
    """
    Inverse of the survival function of the Kumaraswamy distribution.
    """
    _validate_a_b(a, b)
    with mpmath.extradps(5):
        p = _validate_p(p)
        if p == 0:
            return mpmath.mp.one
        if p == 1:
            return mpmath.mp.zero
        a = mpmath.mpf(a)
        b = mpmath.mpf(b)
        return inv_powm1(-mpmath.power(p, 1/b), a)


def mean(a, b):
    """
    Mean of the Kumaraswamy distribution.
    """
    _validate_a_b(a, b)
    with mpmath.extradps(5):
        a = mpmath.mpf(a)
        b = mpmath.mpf(b)
        return b*mpmath.beta(1 + 1/a, b)


def var(a, b):
    """
    Variance of the Kumaraswamy distribution.
    """
    _validate_a_b(a, b)
    with mpmath.extradps(5):
        a = mpmath.mpf(a)
        b = mpmath.mpf(b)
        return b*mpmath.beta(1 + 2/a, b) - mean(a, b)**2


def median(a, b):
    """
    Median of the Kumaraswamy distribution.
    """
    _validate_a_b(a, b)
    with mpmath.extradps(5):
        a = mpmath.mpf(a)
        b = mpmath.mpf(b)
        return inv_powm1(-mpmath.power(0.5, 1/b), a)
