
from mpmath import mp
from mpsci.distributions import fishers_noncentral_hypergeometric as fnchg


def test_basic():
    with mp.workdps(50):
        ntotal = 16
        ngood = 8
        nsample = 10
        nc = mp.mpq(5, 2)
        # Precomputed results using Wolfram:
        #     PDF[FisherHypergeometricDistribution[10, 8, 16, 5/2], k]
        expected_pmf = [
            mp.mpq(64, 433249),
            mp.mpq(2560, 433249),
            mp.mpq(28000, 433249),
            mp.mpq(112000, 433249),
            mp.mpq(175000, 433249),
            mp.mpq(100000, 433249),
            mp.mpq(15625, 433249),
        ]
        sup, pmf = fnchg.support_pmf(nc, ntotal, ngood, nsample)
        assert list(sup) == [2, 3, 4, 5, 6, 7, 8]
        for k in range(len(expected_pmf)):
            assert mp.almosteq(pmf[k], expected_pmf[k])
