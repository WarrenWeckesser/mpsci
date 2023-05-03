import pytest
from mpmath import mp
from mpsci.fun import inv_powm1, pow1pm1, inv_pow1pm1


@pytest.mark.parametrize('y', [2.5, -0.5])
@mp.workdps(40)
def test_inv_powm1(y):
    t = mp.mpf(1.5)
    y = mp.mpf(y)
    x = inv_powm1(t, y)
    assert mp.almosteq(x, mp.power(t + 1, 1/y))


@pytest.mark.parametrize('y', [2.5, -0.5])
@mp.workdps(40)
def test_powm1_inv_powm1_roundtrip(y):
    x = mp.mpf(0.125)
    y = mp.mpf(y)
    t = mp.powm1(x, y)
    x2 = inv_powm1(t, y)
    assert mp.almosteq(x2, x)


@pytest.mark.parametrize('y', [2.5, -0.5])
@mp.workdps(40)
def test_pow1pm1(y):
    x = mp.mpf(0.125)
    y = mp.mpf(y)
    t = pow1pm1(x, y)
    assert mp.almosteq(t, (x+1)**y - 1)


@pytest.mark.parametrize('y', [2.5, -0.5])
@mp.workdps(40)
def test_pow1pm1_inv_pow1pm1_roundtrip(y):
    x = mp.mpf(0.125)
    y = mp.mpf(y)
    t = pow1pm1(x, y)
    x2 = inv_pow1pm1(t, y)
    assert mp.almosteq(x, x2)


@pytest.mark.parametrize('y', [2.5, -0.5])
@mp.workdps(40)
def test_inv_pow1pm1(y):
    t = mp.mpf(2.5)
    y = mp.mpf(y)
    x = inv_pow1pm1(t, y)
    assert mp.almosteq(x, mp.power(t + 1, 1/y,) - 1)
