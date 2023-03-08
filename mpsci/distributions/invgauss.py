"""
Inverse Gaussian distribution
-----------------------------

This implementation uses the same parameterization as the SciPy
implementation in `scipy.stats.invgauss`, except the shape parameter
is called ``m`` instead of ``mu``, to avoid confusion with the common use
of ``mu`` in other parametrizations.  `loc` and `scale` are the standard
location and scale parameters.

A slightly different parametrization is more commonly used (e.g.
the Wikipedia article "Inversed Gaussian distribion" [1]_,
NumPy's `numpy.random.Generator.wald`, Wolfram Alpha's
`InverseGaussianDistribution).  (The parameters `μ` and `λ` of
the Wikipedia article and Wolfram are the same as the ``mean``
and ``scale`` parameters of NumPy's ``wald`` distribution,
respectively.)

To convert from the mpsci parametrization (``m``, ``loc``, ``scale``)
to the more common one, ``loc`` must be 0. Then::

    μ = m*scale
    λ = scale

To go the other way::

    m     = μ/λ
    loc   = 0
    scale = λ

.. [1] "Inverse Gaussian distribution", Wikipedia,
       https://en.wikipedia.org/wiki/Inverse_Gaussian_distribution
"""

from mpmath import mp
from ._common import _validate_p


__all__ = ['pdf', 'logpdf',
           'cdf', 'logcdf', 'invcdf',
           'sf', 'logsf', 'invsf',
           'mean', 'mode', 'var', 'entropy']


def _validate_params(m, loc, scale):
    if m <= 0:
        raise ValueError('m must be positive')
    if scale <= 0:
        raise ValueError('scale must be positive')
    m = mp.mpf(m)
    loc = mp.mpf(loc)
    scale = mp.mpf(scale)
    return m, loc, scale


def pdf(x, m, loc=0, scale=1):
    """
    PDF for the inverse Gaussian distribution.
    """
    with mp.extradps(5):
        m, loc, scale = _validate_params(m, loc, scale)
        x = mp.mpf(x)
        if x <= loc:
            return mp.zero
        z = (x - loc)/scale
        den = scale*mp.sqrt(2*mp.pi*z**3)
        t = ((z - m)/m)**2
        num = mp.exp(-t/(2*z))
        return num/den


def logpdf(x, m, loc=0, scale=1):
    """
    Logarithm of the PDF for the inverse Gaussian distribution.
    """
    with mp.extradps(5):
        m, loc, scale = _validate_params(m, loc, scale)
        x = mp.mpf(x)
        if x <= loc:
            return mp.ninf
        z = (x - loc)/scale
        t = ((z - m)/m)**2
        logp = (-0.5*mp.log(2*mp.pi) - 1.5*mp.log(z)
                - t/(2*z) - mp.log(scale))
        return logp


def cdf(x, m, loc=0, scale=1):
    """
    CDF for the inverse Gaussian distribution.
    """
    with mp.extradps(5):
        m, loc, scale = _validate_params(m, loc, scale)
        x = mp.mpf(x)
        if x <= loc:
            return mp.zero
        z = (x - loc)/scale
        t1 = mp.ncdf((z/m - 1)/mp.sqrt(z))
        t2 = mp.exp(2/m)*mp.ncdf(-(z/m + 1)/mp.sqrt(z))
        return t1 + t2


def logcdf(x, m, loc=0, scale=1):
    """
    Logarithm of the CDF for the inverse Gaussian distribution.
    """
    with mp.extradps(5):
        m, loc, scale = _validate_params(m, loc, scale)
        x = mp.mpf(x)
        if x <= loc:
            return mp.ninf
        z = (x - loc)/scale
        t1 = mp.log(mp.ncdf((z/m - 1)/mp.sqrt(z)))
        t2 = (2/m) + mp.log(mp.ncdf(-(z/m + 1)/mp.sqrt(z)))
        return t1 + mp.log1p(mp.exp(t2 - t1))


def invcdf(p, m, loc=0, scale=1):
    """
    Inverse of the CDF for the inverse Gaussian distribution.
    """
    with mp.extradps(5):
        m, loc, scale = _validate_params(m, loc, scale)
        p = _validate_p(p)
        if p == 0:
            return loc
        if p == 1:
            return mp.inf
        x0 = mode(m, loc, scale)
        # FIXME: This loop assumes convergence!
        while True:
            x1 = x0 + (p - cdf(x0, m, loc, scale))/pdf(x0, m, loc, scale)
            if mp.almosteq(x1, x0):
                break
            x0 = x1
        return x1


def sf(x, m, loc=0, scale=1):
    """
    Survival function for the inverse Gaussian distribution.
    """
    with mp.extradps(5):
        m, loc, scale = _validate_params(m, loc, scale)
        x = mp.mpf(x)
        if x <= loc:
            return mp.one
        z = (x - loc)/scale
        t1 = mp.ncdf(-(z/m - 1)/mp.sqrt(z))
        t2 = mp.exp(2/m)*mp.ncdf(-(z/m + 1)/mp.sqrt(z))
        return t1 - t2


def logsf(x, m, loc=0, scale=1):
    """
    Logarithm of the survival function for the inverse Gaussian distribution.
    """
    with mp.extradps(5):
        m, loc, scale = _validate_params(m, loc, scale)
        x = mp.mpf(x)
        if x <= loc:
            return mp.zero
        z = (x - loc)/scale
        t1 = mp.log(mp.ncdf(-(z/m - 1)/mp.sqrt(z)))
        t2 = 2/m + mp.log(mp.ncdf(-(z/m + 1)/mp.sqrt(z)))
        return t1 + mp.log1p(-mp.exp(t2 - t1))


def invsf(p, m, loc=0, scale=1):
    """
    Inverse of the survival function for the inverse Gaussian distribution.
    """
    with mp.extradps(5):
        m, loc, scale = _validate_params(m, loc, scale)
        p = _validate_p(p)
        if p == 0:
            return mp.inf
        if p == 1:
            return loc
        x0 = mode(m, loc, scale)
        # FIXME: This loop assumes convergence!
        while True:
            x1 = x0 - (p - sf(x0, m, loc, scale))/pdf(x0, m, loc, scale)
            if mp.almosteq(x1, x0):
                break
            x0 = x1
        return x1


def mean(m, loc=0, scale=1):
    """
    Mean of the inverse Gaussian distribution.
    """
    with mp.extradps(5):
        m, loc, scale = _validate_params(m, loc, scale)
        return scale*m + loc


def mode(m, loc=0, scale=1):
    """
    Mode of the inverse Gaussian distribution.
    """
    with mp.extradps(5):
        m, loc, scale = _validate_params(m, loc, scale)
        s = 3*m/2
        # t is equivalent to sqrt(1 + 1/s**2) - 1.
        t = mp.expm1(mp.log1p(1/s**2)/2)
        # mode = m*(sqrt(1 + s**2) - s) = m*s*(sqrt(1 + 1/s**2) - 1) = m*s*t
        return scale*(m*s*t) + loc


def var(m, loc=0, scale=1):
    """
    Variance of the inverse Gaussian distribution.
    """
    with mp.extradps(5):
        m, loc, scale = _validate_params(m, loc, scale)
        return m**3*scale**2


def entropy(m, loc=0, scale=1):
    """
    Differential entropy of the inverse Gaussian distribution.
    """
    with mp.extradps(5):
        m, loc, scale = _validate_params(m, loc, scale)
        t1 = (mp.log(2*mp.pi) + 3*mp.log(m) + 1)/2
        t2 = 3*(mp.exp(2/m) * mp.expint(1, 2/m))/2
        return t1 - t2 + mp.log(scale)
