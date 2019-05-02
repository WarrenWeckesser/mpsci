
import mpmath
from mpsci.distributions import fishers_noncentral_hypergeometric


def test_basic():
    with mpmath.workdps(50):
        ntotal = 16
        ngood = 8
        nsample = 10
        nc = mpmath.mp.mpq(5, 2)
        # Precomputed results using Wolfram:
        #     PDF[FisherHypergeometricDistribution[10, 8, 16, 5/2], k]
        expected_pmf = [
            mpmath.mp.mpq(64, 433249),
            mpmath.mp.mpq(2560, 433249),
            mpmath.mp.mpq(28000, 433249),
            mpmath.mp.mpq(112000, 433249),
            mpmath.mp.mpq(175000, 433249),
            mpmath.mp.mpq(100000, 433249),
            mpmath.mp.mpq(15625, 433249),
        ]
        sup, pmf = fishers_noncentral_hypergeometric.support(nc, ntotal, ngood, nsample)
        assert list(sup) == [2, 3, 4, 5, 6, 7, 8]
        for k in range(len(expected_pmf)):
            assert mpmath.almosteq(pmf[k], expected_pmf[k])
