
import mpmath
import pytest
from mpsci.fun import yeo_johnson, inv_yeo_johnson


mpmath.mp.dps = 60


@pytest.mark.parametrize('x, lmb, expected',
                         [(3, 2, mpmath.mpf('7.5')),
                          (-3, 3, mpmath.mpf('-0.75')),
                          (3, 0, mpmath.log(4)),
                          (-3, 2, -mpmath.log(4))])
def test_yeo_johnson_equality(x, lmb, expected):
    y = yeo_johnson(x, lmb)
    assert mpmath.almosteq(y, expected)
    yinv = inv_yeo_johnson(expected, lmb)
    assert mpmath.almosteq(yinv, x)
