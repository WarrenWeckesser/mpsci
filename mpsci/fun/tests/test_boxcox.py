
import pytest
import mpmath
from mpsci.fun import boxcox, boxcox1p, inv_boxcox, inv_boxcox1p


mpmath.mp.dps = 40


def test_boxcox_lmbda_01():
    for x in [1, 10]:
        assert boxcox(x, 1) == x - 1
        assert boxcox(x, 0) == mpmath.log(x)


def test_boxcox1p_lmbda_01():
    for x in [1, 10]:
        assert boxcox1p(x, 1) == x
        assert boxcox1p(x, 0) == mpmath.log1p(x)


@pytest.mark.parametrize('lmbda', [2.5, -0.5])
def test_boxcox(lmbda):
    x = mpmath.mpf(0.125)
    lmbda = mpmath.mpf(lmbda)
    y = boxcox(x, lmbda)
    assert mpmath.almosteq(y, (x**lmbda - 1)/lmbda)


def test_boxcox_lmbda0():
    x = mpmath.mpf(0.125)
    y = boxcox(x, 0)
    assert mpmath.almosteq(y, mpmath.log(x))


@pytest.mark.parametrize('lmbda', [2.5, -0.5])
def test_boxcox1p(lmbda):
    x = mpmath.mpf(0.125)
    lmbda = mpmath.mpf(lmbda)
    y = boxcox1p(x, lmbda)
    assert mpmath.almosteq(y, ((1 + x)**lmbda - 1)/lmbda)


def test_boxcox1p_lmbda0():
    x = mpmath.mpf(0.125)
    y = boxcox1p(x, 0)
    assert mpmath.almosteq(y, mpmath.log1p(x))


@pytest.mark.parametrize('lmbda', [2.5, -0.5])
def test_boxcox_inv_boxcox_roundtrip(lmbda):
    x = mpmath.mpf('0.125')
    lmbda = mpmath.mpf(lmbda)
    y = boxcox(x, lmbda)
    x2 = inv_boxcox(y, lmbda)
    assert mpmath.almosteq(x2, x)


@pytest.mark.parametrize('lmbda', [2.5, -0.5])
def test_boxcox1p_inv_boxcox1p_roundtrip(lmbda):
    x = mpmath.mpf('0.125')
    lmbda = mpmath.mpf(lmbda)
    y = boxcox1p(x, lmbda)
    x2 = inv_boxcox1p(y, lmbda)
    assert mpmath.almosteq(x2, x)
