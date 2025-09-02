import pytest
from mpmath import mp
from mpsci.stats import mean, var, std, variation, gmean, hmean, pmean


# XXX In some of these tests, equality is asserted even though the
# calculation uses floating point.  That these tests currently pass
# might just be luck.


@mp.workdps(50)
def test_mean():
    assert mean([1, 2, 3]) == 2
    assert mean([1, 2, 3], weights=[3, 1, 3]) == 2


@mp.workdps(50)
def test_var():
    with mp.extraprec(16):
        assert mp.almosteq(var([2, 4, 6]), mp.mpf('8/3'))
        assert mp.almosteq(var([2, 4, 6], ddof=1), 4)


@mp.workdps(50)
def test_std():
    with mp.extraprec(16):
        assert mp.almosteq(std([2, 4, 6]), mp.sqrt(mp.mpf('8/3')))
        assert mp.almosteq(std([2, 4, 6], ddof=1), 2)


@mp.workdps(50)
def test_gmean():
    assert mp.almosteq(gmean([3, 3**3, 3**5]), 27)


@mp.workdps(50)
def test_gmean_with_0():
    assert gmean([3, 1, 0, 9]) == 0


@mp.workdps(50)
def test_gmean_weights_all_one():
    x = [3, 4, 5, 10]
    assert mp.almosteq(gmean(x, weights=[1]*len(x)), gmean(x))


@mp.workdps(50)
def test_gmean_weights():
    x = [2, 3, 5, 8]
    w = [1, 2, 3, 4]
    wgm1 = gmean(x, weights=w)
    wx = sum([[xi]*wi for (xi, wi) in zip(x, w)], [])
    wgm2 = gmean(wx)
    assert mp.almosteq(wgm1, wgm2)


@mp.workdps(50)
def test_gmean_0_weight():
    x = [2, 3, 5, 8, 13, 21]
    w = [1, 3, 5, 0,  4,  0]
    wgm1 = gmean(x, weights=w)
    # Filter out points where the weight is 0.
    xx, ww = zip(*[(xi, wi) for (xi, wi) in zip(x, w) if wi != 0])
    wgm2 = gmean(xx, weights=ww)
    assert mp.almosteq(wgm1, wgm2)


@mp.workdps(50)
def test_hmean():
    with mp.extraprec(16):
        assert mp.almosteq(hmean([1, 2, 16]), mp.mpf('48/25'))


@mp.workdps(50)
def test_pmean():
    with mp.extraprec(16):
        assert mp.almosteq(pmean([3, 4, 5], p=3), 72**mp.mpf('1/3'))
        assert mp.almosteq(pmean([2, 2, 1], p=-2), mp.sqrt(2))
        assert pmean([4, 2, 5, 3], p=mp.inf) == 5
        assert pmean([4, 2, 5, 3], p=-mp.inf) == 2


@mp.workdps(50)
@pytest.mark.parametrize('p', [-2.5, -1, 0, 0.5, 1, 1.75])
def test_pmean_trivial_weights(p):
    x = [3, 4, 10, 1, 1, 25]
    assert mp.almosteq(pmean(x, p=p),
                       pmean(x, p=p, weights=[5]*len(x)))


@mp.workdps(50)
def test_variation0():
    x = [1, 1, 4]
    v = variation(x, ddof=0)
    assert mp.almosteq(v**2, mp.mpf('0.5'))


@mp.workdps(50)
def test_variation1():
    x = [1, 1, 4]
    v = variation(x, ddof=1)
    assert mp.almosteq(v**2, mp.mpf('0.75'))
