
import mpmath
from mpsci.distributions import hypergeometric


def test_basic():
    with mpmath.workdps(50):
        ntotal = 20
        ngood = 14
        nsample = 5
        # Precomputed results using Wolfram
        #   PMF{HypergeometricDistribution[5, 14, 20], k}
        #   CDF{HypergeometricDistribution[5, 14, 20], k}
        expected_pmf = [
            mpmath.mpf('1/2584'),
            mpmath.mpf('35/2584'),
            mpmath.mpf('455/3876'),
            mpmath.mpf('455/1292'),
            mpmath.mpf('1001/2584'),
            mpmath.mpf('1001/7752'),
        ]
        expected_cdf = [
            mpmath.mpf('1/2584'),
            mpmath.mpf('9/646'),
            mpmath.mpf('509/3876'),
            mpmath.mpf('937/1938'),
            mpmath.mpf('6751/7752'),
            mpmath.mpf('1'),
        ]
        for k in range(len(expected_pmf)):
            p = hypergeometric.pmf(k, ntotal, ngood, nsample)
            assert mpmath.almosteq(p, expected_pmf[k])
            c = hypergeometric.cdf(k, ntotal, ngood, nsample)
            assert mpmath.almosteq(c, expected_cdf[k])
            s = hypergeometric.sf(k, ntotal, ngood, nsample)
            assert mpmath.almosteq(s, 1 - expected_cdf[k])
