
import pytest
import mpmath
from mpsci.distributions import beta


mpmath.mp.dps = 40


@pytest.mark.parametrize('a, b', [(1, 1), (4, 2), (3, 5)])
def test_pmf_integer_ab_half(a, b):
    half = mpmath.mpf('0.5')
    p = beta.pdf(half, a, b)
    assert p == mpmath.power(half, a + b - 2) / mpmath.beta(a, b)


@pytest.mark.parametrize('a, b', [(1, 1), (4, 2), (3, 5)])
def test_pmf_integer_ab_fourth(a, b):
    fourth = mpmath.mpf('0.25')
    p = beta.pdf(fourth, a, b)
    expected = (mpmath.power(3, b - 1) / mpmath.power(4, a + b - 2) /
                mpmath.beta(a, b))
    assert mpmath.almosteq(p, expected)
