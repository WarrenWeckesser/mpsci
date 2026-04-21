"""
This module defines hermite_e(n, x), the "probabilist's" Hermite polynomial.
"""

from mpmath import mp


@mp.extradps(5)
def hermite_e(n, x):
    """
    "Probabilist's" Hermite polynomial.
    """
    n = mp.mpf(n)
    x = mp.mpf(x)
    return mp.mpf(2)**(-n/2)*mp.hermite(n, x/mp.sqrt(2))
