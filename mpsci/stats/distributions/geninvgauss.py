"""
Methods for the generalized inverse Gaussian distribution.
"""

import mpmath


# Parameters have been chosen to match the proposed implementation of
# geninvgauss in scipy.

def pdf(x, p, b, loc=0, scale=1):
    """
    Probability density function of the generalized inverse Gaussian
    distribution.

    The PDF for x > loc is

        z**(p - 1) * exp(-b*(z + 1/z)/2))
        ---------------------------------
               scale * K_p(b)

    where z = (x - loc)/scale and K_p(b) is the modified Bessel function of
    the second kind.  For x <= loc, the PDF is zero.
    """
    x = mpmath.mpf(x)
    p = mpmath.mpf(p)
    b = mpmath.mpf(b)
    loc = mpmath.mpf(loc)
    scale = mpmath.mpf(scale)

    if x <= loc:
        return mpmath.mp.zero
    z = (x - loc)/scale
    return (mpmath.power(z, p - 1)
            * mpmath.exp(-b*(z + 1/z)/2)
            / (2*mpmath.besselk(p, b))
            / scale)


def logpdf(x, p, b, loc=0, scale=1):
    """
    Log of the PDF of the generalized inverse Gaussian distribution.

    The PDF for x > loc is

        z**(p - 1) * exp(-b*(z + 1/z)/2))
        ---------------------------------
               scale * K_p(b)

    where z = (x - loc)/scale and K_p(b) is the modified Bessel function of
    the second kind.  For x <= loc, the PDF is zero.
    """
    x = mpmath.mpf(x)
    p = mpmath.mpf(p)
    b = mpmath.mpf(b)
    loc = mpmath.mpf(loc)
    scale = mpmath.mpf(scale)

    if x <= loc:
        return -mpmath.mp.inf
    z = (x - loc)/scale
    return ((p - 1)*mpmath.log(z)
            - b*(z + 1/z)/2
            - mpmath.log(2*mpmath.besselk(p, b))
            - mpmath.log(scale))


def cdf(x, p, b, loc=0, scale=1):
    """
    Cumulative distribution function of the generalized inverse Gaussian
    distribution.

    The CDF is computed by using mpmath.quad to numerically integrate the PDF.
    """
    x = mpmath.mpf(x)
    p = mpmath.mpf(p)
    b = mpmath.mpf(b)
    loc = mpmath.mpf(loc)
    scale = mpmath.mpf(scale)

    if x <= loc:
        return mpmath.mp.zero
    m = mode(p, b, loc, scale)
    # If the mode is in the integration interval, use it to do the integral
    # in two parts.  Otherwise do just one integral.
    if x <= m:
        c = mpmath.quad(lambda t: pdf(t, p, b, loc, scale), [loc, x])
    else:
        c = (mpmath.quad(lambda t: pdf(t, p, b, loc, scale), [loc, m]) +
             mpmath.quad(lambda t: pdf(t, p, b, loc, scale), [m, x]))
    c = min(c, mpmath.mp.one)
    return c


def sf(x, p, b, loc=0, scale=1):
    """
    Survival function of the generalized inverse Gaussian distribution.

    The survival function is computed by using mpmath.quad to numerically
    integrate the PDF.
    """
    x = mpmath.mpf(x)
    p = mpmath.mpf(p)
    b = mpmath.mpf(b)
    loc = mpmath.mpf(loc)
    scale = mpmath.mpf(scale)

    if x <= loc:
        return mpmath.mp.one
    m = mode(p, b, loc, scale)
    # If the mode is in the integration interval, use it to do the integral
    # in two parts.  Otherwise do just one integral.
    if x >= m:
        s = mpmath.quad(lambda t: pdf(t, p, b, loc, scale), [x, mpmath.inf])
    else:
        s = (mpmath.quad(lambda t: pdf(t, p, b, loc, scale), [x, m]) +
             mpmath.quad(lambda t: pdf(t, p, b, loc, scale), [m, mpmath.inf]))
    return s


def mean(p, b, loc=0, scale=1):
    """
    Mean of the generalized inverse Gaussian distribution.

    The mean is
                     K_{p + 1}(b)
        loc + scale --------------
                        K_p(b)

    where K_n(x) is the modified Bessel function of the second kind
    (implemented in mpmath as besselk(n, x)).
    """
    p = mpmath.mpf(p)
    b = mpmath.mpf(b)
    loc = mpmath.mpf(loc)
    scale = mpmath.mpf(scale)

    return loc + scale*mpmath.besselk(p + 1, b)/mpmath.besselk(p, b)


def mode(p, b, loc=0, scale=1):
    """
    Mode of the generalized inverse Gaussian distribution.

    The mode is
                    p - 1 + sqrt((p - 1)**2 + b**2)
        loc + scale -------------------------------
                                  b
    """
    p = mpmath.mpf(p)
    b = mpmath.mpf(b)
    loc = mpmath.mpf(loc)
    scale = mpmath.mpf(scale)

    return loc + scale*(p - 1 + mpmath.sqrt((p - 1)**2 + b**2))/b
