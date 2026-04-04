import pytest
from mpmath import mp
from mpsci.fun import betaincinv


@mp.workdps(120)
def test_betaincinv_bisect():
    a = mp.mpf('0.001')
    b = mp.mpf(2500)
    y = mp.mpf('0.995')
    x = betaincinv(a, b, y, method='bisect')
    # Reference value computed with Wolfram Alpha:
    #     InverseBetaRegularized(995/1000, 1/1000, 2500)
    xref = mp.mpf('1.50151405172215011710526461342152189782949847947012529631'
                  '2822019752091611687647873043898405623312682891958309435281'
                  '9191411e-6')
    assert mp.almosteq(x, xref)


@mp.workdps(120)
@pytest.mark.parametrize('method', ['bisect', 'findroot'])
def test_betaincinv(method):
    a = mp.mpf(5)
    b = mp.mpf(1.5)
    y = mp.mpf(2.0**-20)
    x = betaincinv(a, b, y, method=method)
    # Reference value computed with Wolfram Alpha:
    #     InverseBetaRegularized(2^-20, 5, 3/2)
    xref = mp.mpf('0.05143807008238062389614287680952126503968273616149829970'
                  '1974420720104950942011959551123950357791985719557798519778'
                  '9959446569479')
    assert mp.almosteq(x, xref)
