from mpmath import mp
from mpsci.fun import betaincinv


@mp.workdps(120)
def test_betaincinv():
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
