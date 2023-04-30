
from mpmath import mp
import pytest
from mpsci.stats import yeo_johnson, inv_yeo_johnson


@pytest.mark.parametrize('x, lmb, expected',
                         [(3, 2, '7.5'),
                          (-3, 3, '-0.75')])
@mp.workdps(60)
def test_yeo_johnson_equality(x, lmb, expected):
    expected = mp.mpf(expected)
    y = yeo_johnson(x, lmb)
    assert mp.almosteq(y, expected)
    yinv = inv_yeo_johnson(expected, lmb)
    assert mp.almosteq(yinv, x)


@mp.workdps(60)
def test_yeo_johnson_cases():
    # XXX To do: convert this to a parametrized test.

    x = 3
    lmb = 0
    expected = mp.log(4)
    y = yeo_johnson(x, lmb)
    assert mp.almosteq(y, expected)
    yinv = inv_yeo_johnson(expected, lmb)
    assert mp.almosteq(yinv, x)

    x = -3
    lmb = 2
    expected = -mp.log(4)
    y = yeo_johnson(x, lmb)
    assert mp.almosteq(y, expected)
    yinv = inv_yeo_johnson(expected, lmb)
    assert mp.almosteq(yinv, x)
