
import mpmath
from mpsci.fun import logbinomial

# Note: the tests assert equality of results, even though
# the calculations are floating point.  This works for now, but
# might have to be changed in the future.


mpmath.mp.dps = 50

def test_logbinomial():
    n = 1250
    k = 500
    b1 = logbinomial(n, k)
    b2 = mpmath.log(mpmath.binomial(n, k))
    assert mpmath.almosteq(b1, b2)
