from mpmath import mp


def faddeevaw(z):
    """
    Computes the complex Faddeeva W function.

        W(z) = exp(-z**2) * erfc(-1j * z)

    See https://en.wikipedia.org/wiki/Faddeeva_function.

    Examples
    --------
    >>> from mpmath import mp
    >>> from mpsci.fun import faddeevaw
    >>> mp.dps = 25
    
    >>> z = 3 + 1.5j
    >>> faddeevaw(z)
    mpc(real='0.0832095352862092579274132846', imag='0.1508797901286885260327614389')

    Verify the identity W(-z) = W(z.conjugate()).conjugate().

    >>> faddeevaw(-z), faddeevaw(z.conjugate()).conjugate()
    (mpc(real='-0.08534318299726304595760856355', imag='-0.151844872400130328305182211'),
     mpc(real='-0.08534318299726304595760856355', imag='-0.151844872400130328305182211'))
    """
    z = mp.mpc(z)
    return mp.exp(-z**2) * mp.erfc(-1j*z)


faddeevaw._docstring_re_subs = [
    (r'W\(z\).*z\)', r':math:`W(z) = e^{-z^2}\\textrm{erfc}(-i z)`', 0, 0)
]
