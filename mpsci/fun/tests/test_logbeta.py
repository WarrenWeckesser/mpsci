
from mpmath import mp
from mpsci.fun import logbeta


mp.dps = 50


def test_logbeta_basic():
    lnb = logbeta(250, 350)
    expected = mp.mpf('-409.0878207308227075522631105941596488100197539984096')
    assert mp.almosteq(lnb, expected)
