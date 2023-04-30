
from mpmath import mp
import pytest
from mpsci.stats import yeojohnson, inv_yeojohnson


@pytest.mark.parametrize('x, lmb, expected',
                         [(3, 2, '7.5'),
                          (-3, 3, '-0.75')])
@mp.workdps(60)
def test_yeojohnson_equality(x, lmb, expected):
    expected = mp.mpf(expected)
    y = yeojohnson(x, lmb)
    assert mp.almosteq(y, expected)
    yinv = inv_yeojohnson(expected, lmb)
    assert mp.almosteq(yinv, x)


@mp.workdps(60)
def test_yeojohnson_cases():
    # XXX To do: convert this to a parametrized test.

    x = 3
    lmb = 0
    expected = mp.log(4)
    y = yeojohnson(x, lmb)
    assert mp.almosteq(y, expected)
    yinv = inv_yeojohnson(expected, lmb)
    assert mp.almosteq(yinv, x)

    x = -3
    lmb = 2
    expected = -mp.log(4)
    y = yeojohnson(x, lmb)
    assert mp.almosteq(y, expected)
    yinv = inv_yeojohnson(expected, lmb)
    assert mp.almosteq(yinv, x)
