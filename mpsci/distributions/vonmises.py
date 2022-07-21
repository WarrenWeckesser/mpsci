"""
von Mises distribution
----------------------

The parameters `kappa` and `mu` match those used in the
wikipedia article:

    https://en.wikipedia.org/wiki/Von_Mises_distribution

"""

import mpmath


__all__ = ['pdf', 'logpdf', 'circmean', 'circvar', 'entropy']


def pdf(x, kappa, mu=0):
    """
    Probability density function of the von Mises distribution.
    """
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        kappa = mpmath.mpf(kappa)
        mu = mpmath.mpf(mu)
        i0k = mpmath.besseli(0, kappa)
        numer = mpmath.exp(kappa * mpmath.cos(x - mu))
        return numer / (2*mpmath.pi*i0k)


def logpdf(x, kappa, mu=0):
    """
    Natural logarithm of the PDF of the von Mises distribution.
    """
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        kappa = mpmath.mpf(kappa)
        mu = mpmath.mpf(mu)
        i0k = mpmath.besseli(0, kappa)
        return (kappa * mpmath.cos(x - mu)
                - mpmath.log(2*mpmath.pi) - mpmath.log(i0k))


def circmean(kappa, mu=0):
    """
    Circular mean of the von Mises distribution.

    The circular mean of the distribution is mu.
    """
    return mpmath.mpf(mu)


def circvar(kappa, mu=0):
    """
    Circular variance of the von Mises distribution.

    This is

        1 - I1(kappa)/I0(kappa)

    where I0 and I1 are the modified Bessel functions of the
    first kind.
    """
    with mpmath.extradps(5):
        kappa = mpmath.mpf(kappa)
        mu = mpmath.mpf(mu)
        r = mpmath.besseli(1, kappa) / mpmath.besseli(0, kappa)
        return 1 - r


def entropy(kappa, mu=0):
    """
    Differential entropy of the von Mises distribution.
    """
    with mpmath.extradps(5):
        kappa = mpmath.mpf(kappa)
        mu = mpmath.mpf(mu)
        i0k = mpmath.besseli(0, kappa)
        r = mpmath.besseli(1, kappa) / i0k
        return -kappa * r + mpmath.log(2*mpmath.pi*i0k)
