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
