
import pytest
from mpmath import mp
from mpsci.distributions._common import (
    _validate_p, _validate_moment_n, _validate_x_bounds, _median,
    _find_bracket
)


@pytest.mark.parametrize('p', [-0.25, 1.25])
def test_validate_p_bad_value(p):
    with pytest.raises(ValueError,
                       match=r'p must be in the interval \[0, 1\]'):
        _validate_p(p)


def test_moment_n_bad_value():
    with pytest.raises(ValueError,
                       match='n must be nonnegative'):
        _validate_moment_n(-1)
    with pytest.raises(TypeError,
                       match='n must be an integer'):
        _validate_moment_n(2.25)


def test_validate_x_bounds_bad_low():
    matchstr = r'All values in x must be greater than three \(3\.0\)\.'
    with pytest.raises(ValueError, match=matchstr):
        _validate_x_bounds([2, 4, 8], low=3.0, strict_low=True,
                           lowname='three')


def test_validate_x_bounds_bad_high_with_name():
    matchstr = r'All values in x must be less than three \(3\.0\)\.'
    with pytest.raises(ValueError, match=matchstr):
        _validate_x_bounds([2, 4, 8], high=3.0, strict_high=True,
                           highname='three')


def test_validate_x_bounds_bad_high_without_name():
    matchstr = r'All values in x must be less than 3\.0\.'
    with pytest.raises(ValueError, match=matchstr):
        _validate_x_bounds([2, 4, 8], high=3.0, strict_high=True)


def test_median_even():
    x = [2, 10, 4, 11]
    m = _median(x)
    assert m == 7.0


def test_median_odd():
    x = [2, 10, 4, 5, 11]
    m = _median(x)
    assert m == 5


def test_find_bracket_low_bound():

    def func(x):
        return x**2

    x0, x1 = _find_bracket(func, 0.25, 0.5, 1.0)
    assert (x0, x1) == (0.5, 0.5)


def test_find_bracket_high_bound():

    def func(x):
        return x**2

    x0, x1 = _find_bracket(func, 0.25, 0.0, 0.5)
    assert (x0, x1) == (0.5, 0.5)


def test_find_bracket_a_ninf():

    def func(x):
        return mp.exp(x)

    x0, x1 = _find_bracket(func, 1/16, mp.ninf, 0.0)
    assert x0 < -mp.log(16) < x1
