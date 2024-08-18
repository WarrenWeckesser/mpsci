
import statistics
import pytest
from mpmath import mp
from mpsci.distributions import poisson


def test_support():
    sup = poisson.support(0.25)
    assert next(sup) == 0
    assert next(sup) == 1
    assert next(sup) == 2
    assert 1000 in sup


@pytest.mark.parametrize('lam', [1, 1.5])
def test_pmf(lam):
    with mp.workdps(40):
        with mp.extradps(5):
            u = [(poisson.pmf(i, lam) / mp.exp(-lam) *
                  mp.factorial(i) / mp.power(lam, i)) for i in range(25)]

    assert all([abs(v - mp.one) < 1e-40 for v in u])


@pytest.mark.parametrize('k', [0, 1, 5, 20])
@pytest.mark.parametrize('lam', [1, 1.5])
def test_cdf(k, lam):
    with mp.workdps(40):
        with mp.extradps(5):
            c = poisson.cdf(k, lam)
            S = sum([mp.power(lam, i) / mp.factorial(i)
                    for i in range(k+1)])
            expected = mp.exp(-lam) * S

    assert abs(c - expected) < 1e-40


@pytest.mark.parametrize('k', [0, 1, 5, 20])
@pytest.mark.parametrize('lam', [1, 1.5])
def test_sf(k, lam):
    with mp.workdps(40):
        with mp.extradps(5):
            sf = poisson.sf(k, lam)
            S = sum([mp.power(lam, i) / mp.factorial(i)
                    for i in range(k+1)])
            expected = 1 - mp.exp(-lam) * S

    assert abs(sf - expected) < 1e-40


def test_skewness():
    with mp.workdps(25):
        lam = 16
        sk = poisson.skewness(lam)
        assert mp.almosteq(sk, 0.25)


def test_kurtosis():
    with mp.workdps(25):
        lam = 8
        kurt = poisson.kurtosis(lam)
        assert mp.almosteq(kurt, 0.125)


def test_mle():
    with mp.workdps(40):
        sample = [2.0, 4.0, 8.0, 16.0]
        lam = poisson.mle(sample)
        # For the values in sample, statistics.mean(sample) will give the
        # exact result.
        assert mp.almosteq(lam, statistics.mean(sample))
