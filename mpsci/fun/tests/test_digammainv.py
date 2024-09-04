from mpmath import mp
from mpsci.fun import digammainv


@mp.workdps(50)
def test_roundtrip():
    # Test that digamma(digammainv(y)) == y
    for y in [mp.mpf(-100), mp.mpf('-3.5'), mp.mpf('-0.5'),
              mp.mpf(0), mp.mpf('1e-8'), mp.mpf('0.5'),
              mp.mpf(5000000)]:
        x = digammainv(y)
        assert mp.almosteq(mp.digamma(x), y)


@mp.workdps(50)
def test_recurrence():
    # Test that digammainv(digamma(x) + 1/x) == x + 1
    for x in [mp.mpf('0.25'), mp.mpf(25)]:
        y = digammainv(mp.digamma(x) + 1/x)
        assert mp.almosteq(y, x + 1)


@mp.workdps(50)
def test_gauss_digamma_theorem():
    pi = mp.pi
    r = 3
    m = 5
    y = (-mp.euler - mp.log(2*m) - pi/2 * mp.cot(r*pi/m)
         + 2*sum(mp.cos(2*pi*n*r/m)*mp.log(mp.sin(pi*n/m))
                 for n in range(1, (m - 1)//2 + 1)))
    x = digammainv(y)
    assert mp.almosteq(x, mp.mpf(r)/m)
