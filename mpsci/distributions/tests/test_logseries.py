import pytest
from mpmath import mp
from mpsci.distributions import logseries
from ._utils import call_and_check_mle


def test_support():
    sup = logseries.support(0.25)
    assert next(sup) == 1
    assert next(sup) == 2
    assert next(sup) == 3
    assert 1000 in sup


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


@mp.workdps(50)
def test_skewness():
    sk = logseries.skewness(mp.mpf('4/10'))
    # From WolframAlpha: Skewness[LogSeriesDistribution[4/10]]
    ref = mp.mpf('3.09972644609562838193869934419957172962931406008544')
    assert mp.almosteq(sk, ref)


@mp.workdps(50)
def test_kurtosis():
    kurt = logseries.kurtosis(mp.mpf('4/10'))
    # From WolframAlpha: Kurtosis[LogSeriesDistribution[4/10]] - 3
    ref = mp.mpf('13.6455565131823427023041223124429449608551024593506')
    assert mp.almosteq(kurt, ref)


@pytest.mark.parametrize(
    'x',                                                   # mean
    [[1]*49 + [2],                                         # 1.02
     [1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1],        # 1.2
     [1, 1, 3, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1],     # 1.25
     [1, 1, 2, 3, 4, 3, 4, 3, 2, 1, 4, 1, 4, 2, 4],        # 2.6
     [1, 4, 2, 2, 3, 2, 3, 3, 3, 4],                       # 2.7
     [1, 1, 2, 3, 3, 3, 4, 5],                             # 2.75
     [1, 4, 5, 1, 9, 3, 6, 1, 1, 1, 3, 1],                 # 3.0
     [1, 2, 4, 8, 16],                                     # 6.2
     [14, 1, 22, 29, 1, 55, 32, 1, 1, 16, 1, 13],          # 15.5
     [9, 1, 38, 322, 88, 1, 419, 11755, 1, 253, 103, 15],  # 1083.75
     ]
)
def test_mle(x):
    call_and_check_mle(logseries.mle, logseries.nll, x)
