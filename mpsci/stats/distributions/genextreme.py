"""
Generalized extreme value distribution
--------------------------------------
"""

import mpmath


__all__ = ['pdf', 'cdf']


def pdf(x, xi, mu=0, sigma=1):
    """
    Generalized extreme value distribution probability density function.
    """
    # Formula from wikipedia, which has a sign convention for xi that
    # is the opposite of scipy's shape parameter.
    if xi != 0:
        t = mpmath.power(1 + ((x - mu)/sigma)*xi, -1/xi)
    else:
        t = mpmath.exp(-(x - mu)/sigma)
    p = mpmath.power(t, xi+1) * mpmath.exp(-t) / sigma
    return p


def cdf(x, xi, mu=0, sigma=1):
    """
    Generalized extreme value distribution cumulative density function.
    """
    # Formula from wikipedia, which has a sign convention for xi that
    # is the opposite of scipy's shape parameter.
    if xi != 0:
        t = mpmath.power(1 + ((x - mu)/sigma)*xi, -1/xi)
    else:
        t = mpmath.exp(-(x - mu)/sigma)
    return mpmath.exp(-t)
