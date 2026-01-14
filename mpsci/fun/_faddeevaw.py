import re
from mpmath import mp


def faddeevaw(z):
    """
    Computes the complex Faddeeva W function.

        W(z) = exp(-z**2) * erfc(-1j * z)

    See https://en.wikipedia.org/wiki/Faddeeva_function.
    """
    z = mp.mpc(z)
    return mp.exp(-z**2) * mp.erfc(-1j*z)


faddeevaw._docstring_re_subs = [
    (r'W\(z\).*z\)', r':math:`W(z) = e^{-z^2}\\textrm{erfc}(-i z)`', 0, 0)
]
