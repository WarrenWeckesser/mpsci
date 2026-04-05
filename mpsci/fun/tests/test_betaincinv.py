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


@mp.workdps(110)
def test_betaincinv_findroot_with_initial():
    a = mp.mpf(0.5)
    b = mp.mpf(3)
    y = mp.mpf(2.0**-6)
    initial = mp.mpf('0.000069451')
    x = betaincinv(a, b, y, method=('findroot', initial))
    # Reference value computed with Wolfram Alpha:
    #     InverseBetaRegularized(2^-6, 1/2, 3)
    xref = mp.mpf('0.0000694508753936932716886294902550100334958516845382949'
                  '466052876251357383369396307642834032290316484218510094710'
                  '490268894038624')
    assert mp.almosteq(x, xref)


def test_betaincinv_nan_return():
    assert mp.isnan(betaincinv(0.5, 1.5, 2.5))
    assert mp.isnan(betaincinv(0.5, 1.5, -0.125))


def test_betaincinv_domain_limits():
    assert betaincinv(0.5, 1.5, 0) == 0
    assert betaincinv(0.5, 1.5, 0, complement=True) == 1
    assert betaincinv(0.5, 1.5, 1) == 1
    assert betaincinv(0.5, 1.5, 1, complement=True) == 0


def test_betaincinv_bad_method():
    with pytest.raises(ValueError, match='invalid method'):
        betaincinv(0.5, 1.5, 0.25, method='plate of shrimp')
