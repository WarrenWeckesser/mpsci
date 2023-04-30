
from mpmath import mp
from ..fun import pow1pm1, inv_pow1pm1


__all__ = ['yeojohnson', 'inv_yeojohnson']


def yeojohnson(x, lmbda):
    r"""
    Yeo-Johnson transformation of x.

    See https://en.wikipedia.org/wiki/Power_transform#Yeo%E2%80%93Johnson_transformation

    *See also:* :func:`inv_yeojohnson`
    """
    with mp.extradps(5):
        x = mp.mpf(x)
        lmbda = mp.mpf(lmbda)
        if x >= 0:
            if lmbda == 0:
                return mp.log1p(x)
            else:
                return pow1pm1(x, lmbda) / lmbda
        else:
            if lmbda == 2:
                return -mp.log1p(-x)
            else:
                lmb2 = 2 - lmbda
                return -pow1pm1(-x, lmb2) / lmb2


def inv_yeojohnson(x, lmbda):
    """
    Inverse Yeo-Johnson transformation.

    See https://en.wikipedia.org/wiki/Power_transform#Yeo%E2%80%93Johnson_transformation

    *See also:* :func:`yeojohnson`
    """
    with mp.extradps(5):
        x = mp.mpf(x)
        lmbda = mp.mpf(lmbda)
        if x >= 0:
            if lmbda == 0:
                return mp.expm1(x)
            else:
                return inv_pow1pm1(lmbda*x, lmbda)
        else:
            if lmbda == 2:
                return -mp.expm1(-x)
            else:
                lmb2 = 2 - lmbda
                return -pow1pm1(-lmb2*x, lmb2)
