import mpmath
from mpmath import mp


@mp.extradps(5)
def spherical_besselj(n, z):
    """
    Spherical Bessel function of the first kind.

    https://dlmf.nist.gov/10.47#ii
    """
    n = mp.mpf(n)
    z = mpmath.mpmathify(z)
    return mp.besselj(n + 0.5, z) * mp.sqrt(mp.pi / (2 * z))


@mp.extradps(5)
def spherical_bessely(n, z):
    """
    Spherical Bessel function of the second kind.

    https://dlmf.nist.gov/10.47#ii
    """
    n = mp.mpf(n)
    z = mpmath.mpmathify(z)
    return mp.bessely(n + 0.5, z) * mp.sqrt(mp.pi / (2 * z))


@mp.extradps(5)
def spherical_besseli(n, z):
    """
    Modified spherical Bessel function of the first kind.

    https://dlmf.nist.gov/10.47#ii
    """
    n = mp.mpf(n)
    z = mpmath.mpmathify(z)
    return mp.besseli(n + 0.5, z) * mp.sqrt(mp.pi / (2 * z))


@mp.extradps(5)
def spherical_besselk(n, z):
    """
    Modified spherical Bessel function of the second kind.

    https://dlmf.nist.gov/10.47#ii
    """
    n = mp.mpf(n)
    z = mpmath.mpmathify(z)
    return mp.besselk(n + 0.5, z) * mp.sqrt(mp.pi / (2 * z))
