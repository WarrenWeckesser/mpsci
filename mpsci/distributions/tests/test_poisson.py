
import statistics
import pytest
from mpmath import mp
from mpsci.stats import unique_counts
from mpsci.distributions import poisson
from ._utils import call_and_check_mle


def test_support():
    sup = poisson.support(0.25)
    assert next(sup) == 0
    assert next(sup) == 1
    assert next(sup) == 2
    assert 1000 in sup


@pytest.mark.parametrize('lam', [1, 1.5])
@mp.workdps(40)
def test_pmf(lam):
    with mp.extradps(5):
        u = [(poisson.pmf(i, lam) / mp.exp(-lam) *
              mp.factorial(i) / mp.power(lam, i)) for i in range(25)]

    assert all([abs(v - mp.one) < 1e-40 for v in u])


@pytest.mark.parametrize('k', [0, 1, 5, 20])
@pytest.mark.parametrize('lam', [1, 1.5])
@mp.workdps(40)
def test_cdf(k, lam):
    with mp.extradps(5):
        c = poisson.cdf(k, lam)
        S = sum([mp.power(lam, i) / mp.factorial(i)
                for i in range(k+1)])
        expected = mp.exp(-lam) * S

    assert abs(c - expected) < 1e-40


@pytest.mark.parametrize('k', [0, 1, 5, 20])
@pytest.mark.parametrize('lam', [1, 1.5])
@mp.workdps(40)
def test_sf(k, lam):
    with mp.extradps(5):
        sf = poisson.sf(k, lam)
        S = sum([mp.power(lam, i) / mp.factorial(i)
                 for i in range(k+1)])
        expected = 1 - mp.exp(-lam) * S

    assert abs(sf - expected) < 1e-40


@mp.workdps(25)
def test_skewness():
    lam = 16
    sk = poisson.skewness(lam)
    assert mp.almosteq(sk, 0.25)


@mp.workdps(25)
def test_kurtosis():
    lam = 8
    kurt = poisson.kurtosis(lam)
    assert mp.almosteq(kurt, 0.125)


def test_nll_counts():
    x = [1, 3, 1, 1, 1, 3, 2, 5, 2, 2, 1]
    lam = 0.125
    nll1 = poisson.nll(x, lam)
    xvalues, xcounts = unique_counts(x)
    nll2 = poisson.nll(xvalues, lam, counts=xcounts)
    assert mp.almosteq(nll1, nll2)


@mp.workdps(40)
def test_mle():
    sample = [2.0, 4.0, 8.0, 16.0]
    lam = poisson.mle(sample)
    # For the values in sample, statistics.mean(sample) will give the
    # exact result.
    assert mp.almosteq(lam, statistics.mean(sample))


@pytest.mark.parametrize('x', [(1, 2, 1, 3, 5), (0, 4, 2, 2, 2, 9, 1, 2, 2)])
@mp.workdps(40)
def test_mle_counts(x):
    lam1 = poisson.mle(x)
    xvalues, xcounts = unique_counts(x)
    lam2 = poisson.mle(xvalues, counts=xcounts)
    assert mp.almosteq(lam1, lam2)


@pytest.mark.parametrize('x', [[0, 9, 3, 3, 13, 12, 4, 3, 1],
                               [3, 2, 89, 14, 44, 19]])
@mp.workdps(40)
def test_mle_cases(x):
    call_and_check_mle(poisson.mle, poisson.nll, x)
