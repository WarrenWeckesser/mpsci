"""
Multivariate t Distribution
---------------------------

"""

# Some values from Wolfram Alpha
# PDF[MultivariateTDistribution[{{1, 0},{0,3}}, 3], {3, 5}]
#     = 0.000768318...
# PDF[MultivariateTDistribution[{{1, 1/10},{1/10,3}}, 3], {0, 0}]
#     = 5/(sqrt(299) π)
#     ≈ 0.0920416800862802853500744637173704102495845924002081767282223...
# PDF[MultivariateTDistribution[{{1,1/10,1/25}, {1/10,3,0}, {1/25,0,2}}, 9],
#                               {0,0}]
#     = (3200 sqrt(2/7469))/(189 π^2)
#     ≈ 0.0280719251273608075819567901706183442574259902571887726816304...


from mpmath import mp


__all__ = ['logpdf', 'pdf', 'entropy']

_multivariate = True


def logpdf(x, nu, loc, scale, scale_inv=None):
    """
    Natural logarithm of the PDF for the multivariate t distribution.

    `loc` must be a sequence.  `scale` is the scale matrix; it
    must be an instance of `mpmath.matrix`.  `scale` must be
    positive definite.

    If given, `scale_inv` must be the inverse of `scale`.
    """

    p = mp.mpf(len(loc))
    with mp.extradps(5):
        nu = mp.mpf(nu)
        if scale_inv is None:
            with mp.extradps(5):
                scale_inv = mp.inverse(scale)
        tmp = mp.matrix(scale.cols, 1)
        for k, v in enumerate(loc):
            tmp[k] = mp.mpf(v)
        loc = tmp
        tmp = mp.matrix(scale.cols, 1)
        for k, v in enumerate(x):
            tmp[k] = mp.mpf(v)
        x = tmp
        delta = x - loc
        c = (nu + p)/2
        t1 = -c * mp.log1p((delta.T * scale_inv * delta)[0, 0] / nu)
        t2 = mp.loggamma(c)
        t3 = mp.loggamma(nu/2)
        t4 = (p/2)*mp.log(nu)
        t5 = (p/2)*mp.log(mp.pi)
        with mp.extradps(5):
            det = mp.det(scale)
        t6 = mp.log(det)/2
        return t2 - t3 - t4 - t5 - t6 + t1


def pdf(x, nu, loc, scale, scale_inv=None):
    """
    PDF for the multivariate t distribution.

    `loc` must be a sequence.  `scale` is the scale matrix; it
    must be an instance of `mpmath.matrix`.  `scale` must be
    positive definite.

    If given, `scale_inv` must be the inverse of `scale`.
    """

    return mp.exp(logpdf(x, nu, loc, scale, scale_inv))


def entropy(nu, loc, scale):
    """
    Differential entropy of the multivariate t distribution.

    `loc` must be a sequence.  `scale` is the scale matrix; it must be an
    instance of `mpmath.matrix`.  `scale` must be positive definite; the
    function does not check this.  If `scale` is not positive definite,
    the return value might not be meaningful.
    """
    d = mp.mpf(len(loc))
    with mp.extradps(5):
        nu = mp.mpf(nu)
        loc = [mp.mpf(t) for t in loc]
        mean_nu_d = (nu + d)/2
        half_nu = nu/2
        return (-mp.loggamma(mean_nu_d)
                + mp.loggamma(half_nu)
                + (d/2)*mp.log(nu*mp.pi)
                + mean_nu_d*(mp.digamma(mean_nu_d) - mp.digamma(half_nu))
                + mp.log(abs(mp.det(scale)))/2)
