import pytest
from mpmath import mp
from mpsci.distributions import binomial


@mp.workdps(50)
def test_support():
    n = 5
    p = 0.25
    sup = binomial.support(n, p)
    pmfsum = [binomial.pmf(k, n, p) for k in sup]
    assert mp.almosteq(sum(pmfsum), 1.0)


@pytest.mark.parametrize('cdf_method', ['sumpmf', 'incbeta'])
def test_pmf_cdf_sf_basic(cdf_method):
    with mp.workdps(25):
        p = mp.one/4
        q = mp.one - p
        n = 4
        expected_pmf = [1*q**4,
                        4*p*q**3,
                        6*p**2*q**2,
                        4*p**3*q,
                        1*p**4]
        expected_cdf = mp.zero
        expected_sf = mp.one
        for k in range(len(expected_pmf)):
            # With p=1/4, the following calculations should be exact.
            pmf = binomial.pmf(k, n, p)
            assert pmf == expected_pmf[k]
            cdf = binomial.cdf(k, n, p, method=cdf_method)
            expected_cdf += pmf
            if cdf_method == 'sumpmf':
                assert cdf == expected_cdf
            else:
                assert mp.almosteq(cdf, expected_cdf)
            sf = binomial.sf(k, n, p, method=cdf_method)
            expected_sf -= pmf
            if cdf_method == 'sumpmf':
                assert sf == expected_sf
            else:
                assert mp.almosteq(sf, expected_sf)


@mp.workdps(25)
def test_mean():
    n = 5
    p = mp.one/4
    assert binomial.mean(n, p) == n*p


@mp.workdps(25)
def test_var():
    n = 5
    p = mp.one/4
    assert binomial.var(n, p) == n*p*(1 - p)
