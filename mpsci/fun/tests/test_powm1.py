import pytest
import mpmath
from mpsci.fun import inv_powm1, pow1pm1, inv_pow1pm1


mpmath.mp.dps = 40


@pytest.mark.parametrize('y', [2.5, -0.5])
def test_inv_powm1(y):
    t = mpmath.mpf(1.5)
    y = mpmath.mpf(y)
    x = inv_powm1(t, y)
    assert mpmath.almosteq(x, mpmath.power(t + 1, 1/y))


@pytest.mark.parametrize('y', [2.5, -0.5])
def test_powm1_inv_powm1_roundtrip(y):
    x = mpmath.mpf(0.125)
    y = mpmath.mpf(y)
    t = mpmath.powm1(x, y)
    x2 = inv_powm1(t, y)
    assert mpmath.almosteq(x2, x)


@pytest.mark.parametrize('y', [2.5, -0.5])
def test_pow1pm1(y):
    x = mpmath.mpf(0.125)
    y = mpmath.mpf(y)
    t = pow1pm1(x, y)
    assert mpmath.almosteq(t, (x+1)**y - 1)


@pytest.mark.parametrize('y', [2.5, -0.5])
def test_pow1pm1_inv_pow1pm1_roundtrip(y):
    x = mpmath.mpf(0.125)
    y = mpmath.mpf(y)
    t = pow1pm1(x, y)
    x2 = inv_pow1pm1(t, y)
    assert mpmath.almosteq(x, x2)


@pytest.mark.parametrize('y', [2.5, -0.5])
def test_inv_pow1pm1(y):
    t = mpmath.mpf(2.5)
    y = mpmath.mpf(y)
    x = inv_pow1pm1(t, y)
    assert mpmath.almosteq(x, mpmath.power(t + 1, 1/y,) - 1)
