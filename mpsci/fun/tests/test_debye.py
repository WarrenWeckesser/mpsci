import pytest
from mpmath import mp
from mpsci.fun import debye


@pytest.mark.parametrize('n', [1, 2, 3])
@pytest.mark.parametrize('method', ['quad', 'nsum'])
def test_debye_at_zero(n, method):
    assert debye(0, n=n, method=method) == 1


@pytest.mark.parametrize('x, n', [(0.5, 1), (4, 1), (3, 2)])
@mp.workdps(200)
def test_debye_basic(x, n):
    # We don't have an external source of truth, so for now,
    # test that the two methods ('quad' and 'nsum') give the
    # same result.
    dquad = debye(x, n=n, method='quad')
    dnsum = debye(x, n=n, method='nsum')
    assert mp.almosteq(dquad, dnsum)


def test_debye_bad_n():
    with pytest.raises(ValueError, match='n must be an integer greater than 0'):
        debye(1.25, n=0)


def test_debye_noninteger_n():
    with pytest.raises(ValueError, match='n must be an integer'):
        debye(1.25, n=2.25)


def test_debye_bad_method():
    with pytest.raises(ValueError, match="method must be 'quad' or 'nsum'"):
        debye(1.25, n=3, method='plate of shrimp')
