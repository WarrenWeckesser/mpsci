"""
von Mises distribution
----------------------

The parameters `kappa` and `mu` match those used in the
wikipedia article:

    https://en.wikipedia.org/wiki/Von_Mises_distribution

"""

from mpmath import mp


__all__ = ['pdf', 'logpdf', 'support',  'circmean', 'circvar', 'entropy']


def pdf(x, kappa, mu=0):
    """
    Probability density function of the von Mises distribution.
    """
    with mp.extradps(5):
        x = mp.mpf(x)
        kappa = mp.mpf(kappa)
        mu = mp.mpf(mu)
        i0k = mp.besseli(0, kappa)
        numer = mp.exp(kappa * mp.cos(x - mu))
        return numer / (2*mp.pi*i0k)


def logpdf(x, kappa, mu=0):
    """
    Natural logarithm of the PDF of the von Mises distribution.
    """
    with mp.extradps(5):
        x = mp.mpf(x)
        kappa = mp.mpf(kappa)
        mu = mp.mpf(mu)
        i0k = mp.besseli(0, kappa)
        return (kappa * mp.cos(x - mu) - mp.log(2*mp.pi) - mp.log(i0k))


def circmean(kappa, mu=0):
    """
    Circular mean of the von Mises distribution.

    The circular mean of the distribution is mu.
    """
    return mp.mpf(mu)


def circvar(kappa, mu=0):
    """
    Circular variance of the von Mises distribution.

    This is

        1 - I1(kappa)/I0(kappa)

    where I0 and I1 are the modified Bessel functions of the
    first kind.
    """
    with mp.extradps(5):
        kappa = mp.mpf(kappa)
        mu = mp.mpf(mu)
        r = mp.besseli(1, kappa) / mp.besseli(0, kappa)
        return 1 - r


def support(kappa, mu=0):
    """
    Support of the von Mises distribution.

    The von Mises distribution is periodic, so the PDF is nonzero
    at any x value.  However, the most useful definition of *support*
    for this distribution is the fundamental period of the distribution.
    So this function returns (-pi + mu, pi + mu).  Then, for example,
    the integral of the PDF over the support results in the expected
    value of 1.
    """
    with mp.extradps(5):
        kappa = mp.mpf(kappa)
        mu = mp.mpf(mu)
        return (-mp.pi + mu, mp.pi + mu)


def entropy(kappa, mu=0):
    """
    Differential entropy of the von Mises distribution.
    """
    with mp.extradps(5):
        kappa = mp.mpf(kappa)
        mu = mp.mpf(mu)
        i0k = mp.besseli(0, kappa)
        r = mp.besseli(1, kappa) / i0k
        return -kappa * r + mp.log(2*mp.pi*i0k)
