
import mpmath
from mpsci.fun import boxcox, boxcox1p


# Note: the tests assert equality of results, even though
# the calculations are floating point.  This works for now, but
# might have to be changed in the future.


def test_boxcox():
    for x in [1, 10]:
        assert boxcox(x, 1) == x - 1
        assert boxcox(x, 0) == mpmath.log(x)


def test_boxcox1p():
    for x in [1, 10]:
        assert boxcox1p(x, 1) == x
        assert boxcox1p(x, 0) == mpmath.log(x + 1)
