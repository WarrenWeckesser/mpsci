
import pytest
from mpmath import mp
from mpsci.distributions import hypergeometric


@pytest.mark.parametrize(
    'ntotal, ngood, nsample, expected',
    [(20, 14, 5, range(0, 6)),
     (15, 10, 12, range(7, 11))]
)
def test_support(ntotal, ngood, nsample, expected):
    sup = hypergeometric.support(ntotal, ngood, nsample)
    assert sup == expected


def test_basic():
    with mp.workdps(50):
        ntotal = 20
        ngood = 14
        nsample = 5
        # Precomputed results using Wolfram
        #   PMF{HypergeometricDistribution[5, 14, 20], k}
        #   CDF{HypergeometricDistribution[5, 14, 20], k}
        expected_pmf = [
            mp.mpf('1/2584'),
            mp.mpf('35/2584'),
            mp.mpf('455/3876'),
            mp.mpf('455/1292'),
            mp.mpf('1001/2584'),
            mp.mpf('1001/7752'),
        ]
        expected_cdf = [
            mp.mpf('1/2584'),
            mp.mpf('9/646'),
            mp.mpf('509/3876'),
            mp.mpf('937/1938'),
            mp.mpf('6751/7752'),
            mp.mpf('1'),
        ]
        for k in range(len(expected_pmf)):
            p = hypergeometric.pmf(k, ntotal, ngood, nsample)
            assert mp.almosteq(p, expected_pmf[k])
            c = hypergeometric.cdf(k, ntotal, ngood, nsample)
            assert mp.almosteq(c, expected_cdf[k])
            s = hypergeometric.sf(k, ntotal, ngood, nsample)
            assert mp.almosteq(s, 1 - expected_cdf[k])


@pytest.mark.parametrize('ntotal, ngood, nsample, mean',
                         [(9, 3, 3, 1.0), (16, 4, 6, 1.5)])
def test_mean(ntotal, ngood, nsample, mean):
    # The reference values were computed "by hand" from the formula.
    m = hypergeometric.mean(ntotal, ngood, nsample)
    assert mp.almosteq(m, mean)


@pytest.mark.parametrize('ntotal, ngood, nsample, var',
                         [(9, 3, 3, 0.5), (16, 4, 6, 0.75)])
def test_var(ntotal, ngood, nsample, var):
    # The reference value was computed "by hand" from the formula.
    v = hypergeometric.var(ntotal, ngood, nsample)
    assert mp.almosteq(v, var)
