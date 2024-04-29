
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


@pytest.mark.parametrize('x, lmb, expected_str',
                         [(3, 0, 'mp.log(4)'),
                          (-3, 2, '-mp.log(4)')])
@mp.workdps(60)
def test_yeojohnson_cases(x, lmb, expected_str):
    expected = eval(expected_str)
    y = yeojohnson(x, lmb)
    assert mp.almosteq(y, expected)
    yinv = inv_yeojohnson(expected, lmb)
    assert mp.almosteq(yinv, x)
