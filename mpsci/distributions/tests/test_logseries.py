import pytest
from mpmath import mp
from mpsci.distributions import logseries


@pytest.mark.parametrize('p, k', [(0.5, 3), (0.75, 1), (0.25, 13)])
def test_pmf(p, k):
    with mp.workdps(40):
        p = mp.mpf(p)
        k = mp.mpf(k)
        y = logseries.pmf(k, p)
        assert mp.almosteq(y, -p**k / mp.log1p(-p) / k)


@pytest.mark.parametrize('p, k', [(0.25, 12), (0.975, 1), (0.125, 7)])
def test_cdf_sf_with_pmf_sum(p, k):
    with mp.workdps(40):
        cdf = logseries.cdf(k, p)
        expected_cdf = mp.fsum([logseries.pmf(t, p) for t in range(k+1)])
        assert mp.almosteq(cdf, expected_cdf)
        sf = logseries.sf(k, p)
        assert mp.almosteq(sf, 1 - expected_cdf)


def test_mean():
    with mp.workdps(40):
        p = mp.mpf(0.125)
        mean = logseries.mean(p)
        # From https://en.wikipedia.org/wiki/Logarithmic_distribution
        ref = -p / mp.log1p(-p) / (1 - p)
        assert mp.almosteq(mean, ref)
