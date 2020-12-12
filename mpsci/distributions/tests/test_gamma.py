
import mpmath
from mpsci.distributions import gamma


mpmath.mp.dps = 40


def test_mom():
    x = [1, 2, 3, 4]
    k, theta = gamma.mom(x)
    assert k == mpmath.mpf(5)
    assert theta == mpmath.mpf('0.5')
