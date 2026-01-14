from mpmath import mp


def faddeevaw(z):
    """
    Computes the complex Faddeeva W function.

        W(z) = exp(-z**2) * erfc(-1j * z)

    See https://en.wikipedia.org/wiki/Faddeeva_function.
    """
    z = mp.mpc(z)
    return mp.exp(-z**2) * mp.erfc(-1j*z)
