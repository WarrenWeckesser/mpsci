
import pytest
from mpmath import mp
from mpsci.stats import boxcox, boxcox1p, inv_boxcox, inv_boxcox1p


@mp.workdps(40)
def test_boxcox_lmbda_01():
    for x in [1, 10]:
        assert boxcox(x, 1) == x - 1
        assert boxcox(x, 0) == mp.log(x)


@mp.workdps(40)
def test_boxcox1p_lmbda_01():
    for x in [1, 10]:
        assert boxcox1p(x, 1) == x
        assert boxcox1p(x, 0) == mp.log1p(x)


@pytest.mark.parametrize('lmbda', [2.5, -0.5])
@mp.workdps(40)
def test_boxcox(lmbda):
    x = mp.mpf(0.125)
    lmbda = mp.mpf(lmbda)
    y = boxcox(x, lmbda)
    assert mp.almosteq(y, (x**lmbda - 1)/lmbda)


@mp.workdps(40)
def test_boxcox_lmbda0():
    x = mp.mpf(0.125)
    y = boxcox(x, 0)
    assert mp.almosteq(y, mp.log(x))


@pytest.mark.parametrize('lmbda', [2.5, -0.5])
@mp.workdps(40)
def test_boxcox1p(lmbda):
    x = mp.mpf(0.125)
    lmbda = mp.mpf(lmbda)
    y = boxcox1p(x, lmbda)
    assert mp.almosteq(y, ((1 + x)**lmbda - 1)/lmbda)


@mp.workdps(40)
def test_boxcox1p_lmbda0():
    x = mp.mpf(0.125)
    y = boxcox1p(x, 0)
    assert mp.almosteq(y, mp.log1p(x))


@pytest.mark.parametrize('lmbda', [0, 2.5, -0.5])
@mp.workdps(40)
def test_boxcox_inv_boxcox_roundtrip(lmbda):
    x = mp.mpf('0.125')
    lmbda = mp.mpf(lmbda)
    y = boxcox(x, lmbda)
    x2 = inv_boxcox(y, lmbda)
    assert mp.almosteq(x2, x)


@pytest.mark.parametrize('lmbda', [0, 2.5, -0.5])
@mp.workdps(40)
def test_boxcox1p_inv_boxcox1p_roundtrip(lmbda):
    x = mp.mpf('0.125')
    lmbda = mp.mpf(lmbda)
    y = boxcox1p(x, lmbda)
    x2 = inv_boxcox1p(y, lmbda)
    assert mp.almosteq(x2, x)
