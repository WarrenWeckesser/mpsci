
from mpmath import mp
from mpsci.distributions import gauss_kuzmin


def test_support():
    sup = gauss_kuzmin.support()
    assert next(sup) == 1
    assert next(sup) == 2
    assert next(sup) == 3
    assert 1000 in sup


@mp.workdps(40)
def test_pmf():
    one = mp.one
    for k in [1, 2, 3, 9]:
        p = gauss_kuzmin.pmf(k)
        expected = -mp.log1p(-one/(k + one)**2)/mp.log(2)
        assert mp.almosteq(p, expected)


@mp.workdps(40)
def test_logpmf():
    one = mp.one
    for k in [1, 2, 3, 9]:
        logp = gauss_kuzmin.logpmf(k)
        expected = mp.log(-mp.log1p(-one/(k + one)**2)/mp.log(2))
        assert mp.almosteq(logp, expected)


@mp.workdps(40)
def test_cdf():
    one = mp.one
    two = mp.mpf(2)
    for k in [1, 2, 3, 9]:
        p = gauss_kuzmin.cdf(k)
        assert mp.almosteq(p, 1 - mp.log((k + two)/(k + one), b=2))


@mp.workdps(40)
def test_cdf_invcdf_roundtrip():
    for k in [1, 2, 3, 9]:
        p = gauss_kuzmin.cdf(k)
        k2 = gauss_kuzmin.invcdf(p)
        assert mp.almosteq(k2, k)


@mp.workdps(40)
def test_sf():
    one = mp.one
    two = mp.mpf(2)
    for k in [1, 2, 3, 9]:
        p = gauss_kuzmin.sf(k)
        assert mp.almosteq(p, mp.log((k + two)/(k + one), b=2))


@mp.workdps(40)
def test_sf_invsf_roundtrip():
    for k in [1, 2, 3, 9]:
        p = gauss_kuzmin.sf(k)
        k2 = gauss_kuzmin.invsf(p)
        assert mp.almosteq(k2, k)


@mp.workdps(40)
def test_median():
    median = gauss_kuzmin.median()
    assert median == 2


@mp.workdps(40)
def test_mode():
    mode = gauss_kuzmin.mode()
    assert mode == 1
