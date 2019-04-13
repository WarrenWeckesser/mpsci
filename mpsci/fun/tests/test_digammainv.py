
import mpmath
from mpsci.fun import digammainv


def test_roundtrip():
    # Test that digamma(digammainv(y)) == y
    with mpmath.workdps(50):
        for y in [mpmath.mpf(-100), mpmath.mpf('-3.5'), mpmath.mpf('-0.5'),
                  mpmath.mpf(0), mpmath.mpf('1e-8'), mpmath.mpf('0.5'),
                  mpmath.mpf(5000000)]:
            x = digammainv(y)
            assert mpmath.almosteq(mpmath.digamma(x), y)


def test_recurrence():
    # Test that digammainv(digamma(x) + 1/x) == x + 1
    with mpmath.workdps(50):
        for x in [mpmath.mpf('0.25'), mpmath.mpf(25)]:
            y = digammainv(mpmath.digamma(x) + 1/x)
            assert mpmath.almosteq(y, x + 1)


def test_gauss_digamma_theorem():
    pi = mpmath.pi
    r = 3
    m = 5
    y = (-mpmath.euler - mpmath.log(2*m) - pi/2 * mpmath.cot(r*pi/m)
         + 2*sum(mpmath.cos(2*pi*n*r/m)*mpmath.log(mpmath.sin(pi*n/m))
                 for n in range(1, (m - 1)//2 + 1)))
    x = digammainv(y)
    assert mpmath.almosteq(x, mpmath.mpf(r)/m)
