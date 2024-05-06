import mpmath
from mpmath import mp


def _wright_bessel_term(k, z, rho, beta):
    numer = mp.power(z, k)
    try:
        g = mp.gamma(k*rho + beta)
    except ValueError:
        return mp.zero
    denom = g*mp.factorial(k)
    return numer / denom


def wright_bessel(z, rho, beta):
    """
    Wright's generalized Bessel function.

    The parameter naming follows

        https://dlmf.nist.gov/10.46

    but here the `z` parameter is given first.

    z and beta can be complex.
    rho must be real and greater than -1.

    See also:

    * https://appliedmath.brown.edu/sites/default/files/fractional/36%20TheWrightFunctions.pdf
    * http://arxiv.org/pdf/2304.02903
    * http://arxiv.org/pdf/2306.11381
    * https://en.wikipedia.org/wiki/Bessel%E2%80%93Maitland_function

    """
    with mp.extradps(5):
        z = mp.mpmathify(z)
        rho = mp.mpf(rho)
        if rho <= -1:
            raise ValueError('rho must be greater than -1')
        beta = mp.mpmathify(beta)

        try:
            g = mp.gamma(beta)
        except ValueError:
            g = mp.inf

        # Handle special cases.
        if z == 0:
            return 1 / g
        if rho == 0:
            return mp.exp(z) / g
        if rho == 1:
            rootz = mp.sqrt(z)
            return (rootz**((1 - beta)) *
                    mpmath.besseli(beta - 1, 2*rootz))

        # Nothing special, so just sum the series.
        return mp.nsum(lambda k: _wright_bessel_term(k, z, rho, beta),
                       [0, mp.inf])
