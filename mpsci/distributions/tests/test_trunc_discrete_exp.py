import pytest
from mpmath import mp
from mpsci.distributions import trunc_discrete_exp


@pytest.mark.parametrize('lam, n', [(0.25, 8), (3.0, 7), (0.0, 5)])
@mp.workdps(50)
def test_cdf_sf(lam, n):
    for k in trunc_discrete_exp.support(lam, n):
        c = trunc_discrete_exp.cdf(k, lam, n)
        computed_c = mp.fsum([trunc_discrete_exp.pmf(i, lam, n)
                              for i in range(0, k + 1)])
        assert mp.almosteq(c, computed_c)
        s = trunc_discrete_exp.sf(k, lam, n)
        computed_s = mp.fsum([trunc_discrete_exp.pmf(i, lam, n)
                              for i in range(k + 1, n)])
        assert mp.almosteq(s, computed_s)


@pytest.mark.parametrize('lam, n', [(0.25, 63), (0.001, 100), (0, 5)])
@mp.workdps(50)
def test_mean(lam, n):
    m = trunc_discrete_exp.mean(lam, n)
    computed_m = mp.fsum([k*trunc_discrete_exp.pmf(k, lam, n)
                          for k in trunc_discrete_exp.support(lam, n)])
    assert mp.almosteq(m, computed_m)


@pytest.mark.parametrize('lam, n', [(0.25, 63), (0.001, 100), (0, 5)])
@mp.workdps(50)
def test_var(lam, n):
    v = trunc_discrete_exp.var(lam, n)
    m = trunc_discrete_exp.mean(lam, n)
    computed_v = mp.fsum([(k - m)**2*trunc_discrete_exp.pmf(k, lam, n)
                          for k in trunc_discrete_exp.support(lam, n)])
    assert mp.almosteq(v, computed_v)


@pytest.mark.parametrize('lam, n', [(0.25, 63), (0.001, 3000), (0, 5)])
@mp.workdps(50)
def test_entropy(lam, n):
    h = trunc_discrete_exp.entropy(lam, n)
    pmf = [trunc_discrete_exp.pmf(k, lam, n)
           for k in trunc_discrete_exp.support(lam, n)]
    computed_h = -mp.fsum([mp.log(p)*p for p in pmf])
    assert mp.almosteq(h, computed_h)
