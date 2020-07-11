"""
Generalized exponential distribution
------------------------------------

This is the same distribution as `scipy.stats.genexpon`.
"""

import mpmath


__all__ = ['pdf', 'logpdf', 'cdf', 'sf', 'invcdf']


def pdf(x, a, b, c, loc=0, scale=1):
    """
    PDF of the generalized exponential distribution.
    """
    # XXX Validate signs of args, check for x < loc, etc.

    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        a = mpmath.mpf(a)
        b = mpmath.mpf(b)
        c = mpmath.mpf(c)
        loc = mpmath.mpf(loc)
        scale = mpmath.mpf(scale)
        z = (x - loc) / scale
        s = a + b
        r = b / c
        p = ((a + b*(-mpmath.expm1(-c*z))) *
             mpmath.exp(-s*z + r*(-mpmath.expm1(-c*z))))
    return p


def logpdf(x, a, b, c, loc=0, scale=1):
    """
    Natural logarithm of the PDF of the generalized exponential distribution.
    """
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        a = mpmath.mpf(a)
        b = mpmath.mpf(b)
        c = mpmath.mpf(c)
        loc = mpmath.mpf(loc)
        scale = mpmath.mpf(scale)
        z = (x - loc) / scale
        s = a + b
        r = b / c
        logp = (mpmath.log(a + b*(-mpmath.expm1(-c*z))) +
                (-s*z + r*(-mpmath.expm1(-c*z))))
    return logp


def cdf(x, a, b, c, loc=0, scale=1):
    """
    CDF of the generalized exponential distribution.
    """
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        a = mpmath.mpf(a)
        b = mpmath.mpf(b)
        c = mpmath.mpf(c)
        loc = mpmath.mpf(loc)
        scale = mpmath.mpf(scale)
        z = (x - loc) / scale
        s = a + b
        r = b / c
        result = -mpmath.expm1(-s*z + r*(-mpmath.expm1(-c*z)))
    return result


def sf(x, a, b, c, loc=0, scale=1):
    """
    Survival function of the generalized exponential distribution.
    """
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        a = mpmath.mpf(a)
        b = mpmath.mpf(b)
        c = mpmath.mpf(c)
        loc = mpmath.mpf(loc)
        scale = mpmath.mpf(scale)
        z = (x - loc) / scale
        s = a + b
        r = b / c
        result = mpmath.exp(-s*z + r*(-mpmath.expm1(-c*z)))
    return result


def invcdf(p, a, b, c, loc=0, scale=1):
    with mpmath.extradps(5):
        p = mpmath.mpf(p)
        a = mpmath.mpf(a)
        b = mpmath.mpf(b)
        c = mpmath.mpf(c)
        loc = mpmath.mpf(loc)
        scale = mpmath.mpf(scale)
        s = a + b
        r = b / c

        y = -mpmath.log1p(-p)
        s = a + b
        r = b / c

        def _genexpon_invcdf_rootfunc(z):
            return s*z + r*mpmath.expm1(-c*z) - y

        z0 = y / s
        z1 = (y + r) / s
        z = mpmath.findroot(_genexpon_invcdf_rootfunc,
                            (z0, z1), solver='anderson')
        x = loc + scale*z
    return x
