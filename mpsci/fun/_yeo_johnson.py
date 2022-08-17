
import mpmath
from ._powm1 import pow1pm1, inv_pow1pm1


__all__ = ['yeo_johnson', 'inv_yeo_johnson']


def yeo_johnson(x, lmbda):
    r"""
    Yeo-Johnson transformation of x.

    See https://en.wikipedia.org/wiki/Power_transform#Yeo%E2%80%93Johnson_transformation

    *See also:* :func:`inv_yeo_johnson`
    """
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        lmbda = mpmath.mpf(lmbda)
        if x >= 0:
            if lmbda == 0:
                return mpmath.log1p(x)
            else:
                return pow1pm1(x, lmbda) / lmbda
        else:
            if lmbda == 2:
                return -mpmath.log1p(-x)
            else:
                lmb2 = 2 - lmbda
                return -pow1pm1(-x, lmb2) / lmb2


def inv_yeo_johnson(x, lmbda):
    """
    Inverse Yeo-Johnson transformation.

    See https://en.wikipedia.org/wiki/Power_transform#Yeo%E2%80%93Johnson_transformation

    *See also:* :func:`yeo_johnson`
    """
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        lmbda = mpmath.mpf(lmbda)
        if x >= 0:
            if lmbda == 0:
                return mpmath.expm1(x)
            else:
                return inv_pow1pm1(lmbda*x, lmbda)
        else:
            if lmbda == 2:
                return -mpmath.expm1(-x)
            else:
                lmb2 = 2 - lmbda
                return -pow1pm1(-lmb2*x, lmb2)
