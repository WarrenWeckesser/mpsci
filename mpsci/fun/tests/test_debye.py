import pytest
from mpmath import mp
from mpsci.fun import debye


@pytest.mark.parametrize('n', [1, 2, 3])
@pytest.mark.parametrize('method', ['quad', 'nsum'])
def test_debye_at_zero(n, method):
    assert debye(0, n=n, method=method) == 1


@pytest.mark.parametrize('x, n', [(0.5, 1), (4, 1), (3, 2)])
@mp.workdps(200)
def test_debye_method_consistency(x, n):
    # Test that the two methods ('quad' and 'nsum') give the same result.
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


@mp.workdps(50)
@pytest.mark.parametrize('method', ['quad', 'nsum'])
def test_debye_against_wolfram_alpha(method):
    n = 3
    x = mp.mpf(1)
    d = debye(x, n=n, method=method)
    # This integral was computed with Wolfram Alpha (https://www.wolframalpha.com):
    #    Integral x^3/(exp(x) - 1) from 0 to 1
    intgrl = mp.mpf('0.2248051880259382266998728764395876663794981679095304042')
    ref = (3/x**3) * intgrl
    assert mp.almosteq(d, ref)
