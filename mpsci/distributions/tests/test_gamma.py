
import mpmath
from mpsci.distributions import gamma


mpmath.mp.dps = 40


# This is the value of CDF[GammaDistribution[2, 3], 1] computed with
# Wolfram Alpha
_cdf123str = ("0.04462491923494766609919453743282711006251725357136112381217"
              "305716359456368679682785494521477976673845")

def test_cdf():
    c = gamma.cdf(1, k=2, theta=3)
    expected = mpmath.mpf(_cdf123str)
    assert mpmath.almosteq(c, expected)


def test_sf():
    s = gamma.sf(1, k=2, theta=3)
    expected = 1 - mpmath.mpf(_cdf123str)
    assert mpmath.almosteq(s, expected)


def test_mom():
    x = [1, 2, 3, 4]
    k, theta = gamma.mom(x)
    assert k == mpmath.mpf(5)
    assert theta == mpmath.mpf('0.5')
