
import mpmath
from mpsci.fun import linspace


def test_basic():
    a = mpmath.mpf('1.0')
    b = mpmath.mpf('3.0')
    vals = linspace(a, b, 3)
    assert vals[0] == a
    assert vals[2] == b
    assert vals[1] == 2


def test_more_basic():
    a = mpmath.mpf(0)
    b = mpmath.mpf(-2)
    vals = linspace(a, b, 9)
    for i in range(9):
        assert vals[i] == -i/4
