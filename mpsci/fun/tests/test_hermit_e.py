import pytest
from mpmath import mp
from mpsci.fun import hermite_e


@pytest.mark.parametrize('x', [-7, -2.5, 0.125, 1.25, 4.75])
@mp.workdps(25)
def test_hermite_e_n3(x):
    y = hermite_e(3, x)
    assert mp.almosteq(y, mp.polyval([0, -3, 0, 1], x, asc=True))


@pytest.mark.parametrize('x', [-7, -2.5, 0.125, 1.25, 4.75])
@mp.workdps(25)
def test_hermite_e_n8(x):
    y = hermite_e(8, x)
    assert mp.almosteq(y, mp.polyval([105, 0, -420, 0, 210, 0, -28, 0, 1],
                                     x, asc=True))
