"""
This module defines functions for the noncentral chi-square distribution.
"""

import mpmath


def pdf(x, k, lam):
    """
    PDF for the noncentral chi-square distribution.
    """
    x = mpmath.mpf(x)
    k = mpmath.mpf(k)
    lam = mpmath.mpf(lam)
    p = (mpmath.exp(-(x + lam)/2) * mpmath.power(x / lam, (k/2 - 1)/2) *
         mpmath.besseli(k/2 - 1, mpmath.sqrt(lam*x))/2)
    return p
