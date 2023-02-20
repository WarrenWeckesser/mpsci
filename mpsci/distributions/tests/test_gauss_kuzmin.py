
from mpmath import mp
from mpsci.distributions import gauss_kuzmin


def test_pmf():
    with mp.workdps(40):
        one = mp.one
        for k in [1, 2, 3, 9]:
            p = gauss_kuzmin.pmf(k)
            expected = -mp.log1p(-one/(k + one)**2)/mp.log(2)
            assert mp.almosteq(p, expected)


def test_logpmf():
    with mp.workdps(40):
        one = mp.one
        for k in [1, 2, 3, 9]:
            logp = gauss_kuzmin.logpmf(k)
            expected = mp.log(-mp.log1p(-one/(k + one)**2)/mp.log(2))
            assert mp.almosteq(logp, expected)


def test_cdf():
    with mp.workdps(40):
        one = mp.one
        two = mp.mpf(2)
        for k in [1, 2, 3, 9]:
            p = gauss_kuzmin.cdf(k)
            assert mp.almosteq(p, 1 - mp.log((k + two)/(k + one), b=2))


def test_cdf_invcdf_roundtrip():
    with mp.workdps(40):
        for k in [1, 2, 3, 9]:
            p = gauss_kuzmin.cdf(k)
            k2 = gauss_kuzmin.invcdf(p)
            assert mp.almosteq(k2, k)


def test_sf():
    with mp.workdps(40):
        one = mp.one
        two = mp.mpf(2)
        for k in [1, 2, 3, 9]:
            p = gauss_kuzmin.sf(k)
            assert mp.almosteq(p, mp.log((k + two)/(k + one), b=2))


def test_sf_invsf_roundtrip():
    with mp.workdps(40):
        for k in [1, 2, 3, 9]:
            p = gauss_kuzmin.sf(k)
            k2 = gauss_kuzmin.invsf(p)
            assert mp.almosteq(k2, k)


def test_median():
    with mp.workdps(40):
        median = gauss_kuzmin.median()
        assert median == 2


def test_mode():
    with mp.workdps(40):
        mode = gauss_kuzmin.mode()
        assert mode == 1
