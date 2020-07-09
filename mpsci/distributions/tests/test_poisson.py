
import pytest
import mpmath
from mpsci.distributions import poisson


mpmath.mp.dps = 40


@pytest.mark.parametrize('lam', [mpmath.mp.one, mpmath.mpf('1.5')])
def test_pmf(lam):
    with mpmath.extradps(5):
        u = [(poisson.pmf(i, lam) / mpmath.exp(-lam) *
              mpmath.factorial(i) / mpmath.power(lam, i)) for i in range(25)]

    assert all([abs(v - mpmath.mp.one) < 1e-40 for v in u])


@pytest.mark.parametrize('k', [0, 1, 5, 20])
@pytest.mark.parametrize('lam', [mpmath.mp.one, mpmath.mpf('1.5')])
def test_cdf(k, lam):
    with mpmath.extradps(5):
        c = poisson.cdf(k, lam)
        S = sum([mpmath.power(lam, i) / mpmath.factorial(i)
                 for i in range(k+1)])
        expected = mpmath.exp(-lam) * S

    assert abs(c - expected) < 1e-40


@pytest.mark.parametrize('k', [0, 1, 5, 20])
@pytest.mark.parametrize('lam', [mpmath.mp.one, mpmath.mpf('1.5')])
def test_sf(k, lam):
    with mpmath.extradps(5):
        sf = poisson.sf(k, lam)
        S = sum([mpmath.power(lam, i) / mpmath.factorial(i)
                 for i in range(k+1)])
        expected = 1 - mpmath.exp(-lam) * S

    assert abs(sf - expected) < 1e-40
