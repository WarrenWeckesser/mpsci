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


import mpmath


__all__ = ['logpdf', 'pdf']

_multivariate = True


def logpdf(x, nu, loc, scale, scale_inv=None):
    """
    Natural logarithm of the PDF for the multivariate t distribution.

    `loc` must be a sequence.  `scale` is the scale matrix; it
    must be an instance of `mpmath.matrix`.  `scale` must be
    positive definite.

    If given, `scale_inv` must be the inverse of `scale`.
    """

    p = mpmath.mpf(len(loc))
    with mpmath.extradps(5):
        nu = mpmath.mpf(nu)
        if scale_inv is None:
            with mpmath.extradps(5):
                scale_inv = mpmath.inverse(scale)
        tmp = mpmath.matrix(scale.cols, 1)
        for k, v in enumerate(loc):
            tmp[k] = mpmath.mpf(v)
        loc = tmp
        tmp = mpmath.matrix(scale.cols, 1)
        for k, v in enumerate(x):
            tmp[k] = mpmath.mpf(v)
        x = tmp
        delta = x - loc
        c = (nu + p)/2
        t1 = -c * mpmath.log1p((delta.T * scale_inv * delta)[0, 0] / nu)
        t2 = mpmath.loggamma(c)
        t3 = mpmath.loggamma(nu/2)
        t4 = (p/2)*mpmath.log(nu)
        t5 = (p/2)*mpmath.log(mpmath.pi)
        with mpmath.extradps(5):
            det = mpmath.det(scale)
        t6 = mpmath.log(det)/2
        return t2 - t3 - t4 - t5 - t6 + t1


def pdf(x, nu, loc, scale, scale_inv=None):
    """
    PDF for the multivariate t distribution.

    `loc` must be a sequence.  `scale` is the scale matrix; it
    must be an instance of `mpmath.matrix`.  `scale` must be
    positive definite.

    If given, `scale_inv` must be the inverse of `scale`.
    """

    return mpmath.exp(logpdf(x, nu, loc, scale, scale_inv))
