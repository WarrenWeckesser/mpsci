import pytest
from mpmath import mp
from mpsci.distributions import negative_hypergeometric, hypergeometric


@pytest.mark.parametrize(
    'k, ntotal, ngood, untilnbad',
    [(3, 20, 10, 5),
     (0, 44, 12, 29),
     (22, 44, 23, 19)]
)
@mp.workdps(40)
def test_cdf_hypergeom_relation(k, ntotal, ngood, untilnbad):
    # Test the relation between the CDFs of the negative hypergeometric
    # distribution and the hypergeometric distribution.
    hg_ngood = ntotal - ngood
    hg_nsample = k + untilnbad
    hg_k = untilnbad - 1
    cdf = negative_hypergeometric.cdf(k, ntotal, ngood, untilnbad)
    val = 1 - hypergeometric.cdf(hg_k, ntotal, hg_ngood, hg_nsample)
    assert mp.almosteq(cdf, val)
