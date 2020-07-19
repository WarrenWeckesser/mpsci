
import mpmath
from mpsci.stats import mean, var, std, variation, gmean, hmean, pmean


# XXX In some of these tests, equality is asserted even though the
# calculation uses floating point.  That these tests currently pass
# might just be luck.


def test_mean():
    assert mean([1, 2, 3]) == 2
    assert mean([1, 2, 3], weights=[3, 1, 3]) == 2


def test_var():
    with mpmath.extraprec(16):
        assert mpmath.almosteq(var([2, 4, 6]), mpmath.mpf('8/3'))
        assert mpmath.almosteq(var([2, 4, 6], ddof=1), 4)


def test_std():
    with mpmath.extraprec(16):
        assert mpmath.almosteq(std([2, 4, 6]), mpmath.sqrt(mpmath.mpf('8/3')))
        assert mpmath.almosteq(std([2, 4, 6], ddof=1), 2)


def test_gmean():
    assert gmean([3, 3**3, 3**5]) == 27


def test_hmean():
    with mpmath.extraprec(16):
        assert mpmath.almosteq(hmean([1, 2, 16]), mpmath.mpf('48/25'))


def test_pmean():
    with mpmath.extraprec(16):
        assert mpmath.almosteq(pmean([3, 4, 5], 3), 72**mpmath.mpf('1/3'))
        assert mpmath.almosteq(pmean([2, 2, 1], -2), mpmath.sqrt(2))
        assert pmean([4, 2, 5, 3], mpmath.inf) == 5
        assert pmean([4, 2, 5, 3], -mpmath.inf) == 2


def test_variation0():
    x = [1, 1, 4]
    v = variation(x, ddof=0)
    assert mpmath.almosteq(v**2, mpmath.mpf('0.5'))


def test_variation1():
    x = [1, 1, 4]
    v = variation(x, ddof=1)
    assert mpmath.almosteq(v**2, mpmath.mpf('0.75'))
