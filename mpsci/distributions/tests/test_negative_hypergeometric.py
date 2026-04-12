from itertools import accumulate
import pytest
from mpmath import mp
from mpsci.distributions import negative_hypergeometric, hypergeometric
from mpsci.stats import mean


def test_support():
    ngood = 10
    sup = negative_hypergeometric.support(20, ngood, 8)
    assert sup == range(ngood + 1)


@mp.workdps(20)
def test_pmf_against_r():
    # The reference values were computed with the function dnhyper(x, n, m, r) from
    # the R extraDistr package.  That packagr uses a different parameterization.
    # To compute pmf(k, ntotal, ngood, nuntilbad), the translation to R is
    #     x = k + nuntilbad
    #     n = ngood
    #     m = ntotal - ngood
    #     r = nuntilbad
    # In R:
    # > library(extraDistr)
    # > options(digits=13)
    # > ntotal = 11
    # > ngood = 6
    # > untilnbad = 4
    # > k = seq(0, 6)
    # > x = k + untilnbad
    # > n = ngood
    # > m = ntotal - ngood
    # > r = untilnbad
    # > dnhyper(x, n, m, r)
    # [1] 0.01515151515152 0.05194805194805 0.10822510822511 0.17316017316017
    # [5] 0.22727272727273 0.24242424242424 0.18181818181818
    # > dnhyper(x, n, m, r, log=TRUE)
    # [1] -4.189654742026 -2.957511060734 -2.223541885654 -1.753538256408
    # [5] -1.481604540924 -1.417066019787 -1.704748092238
    ntotal = 11
    ngood = 6
    untilnbad = 4
    sup, pmf = negative_hypergeometric.support_pmf(ntotal, ngood, untilnbad)
    ref = [0.01515151515152, 0.05194805194805, 0.10822510822511, 0.17316017316017,
           0.22727272727273, 0.24242424242424, 0.18181818181818]
    for k in sup:
        assert mp.almosteq(pmf[k], ref[k], rel_eps=1e-8, abs_eps=0)


def test_pmf_outside_support():
    ntotal = 443
    ngood = 60
    untilnbad = 10
    sup = negative_hypergeometric.support(ntotal, ngood, untilnbad)
    assert negative_hypergeometric.pmf(sup[0] - 1, ntotal, ngood, untilnbad) == 0
    assert negative_hypergeometric.pmf(sup[-1] + 1, ntotal, ngood, untilnbad) == 0


def test_logpmf_outside_support():
    ntotal = 231
    ngood = 200
    untilnbad = 10
    sup = negative_hypergeometric.support(ntotal, ngood, untilnbad)
    assert (negative_hypergeometric.logpmf(sup[0] - 1, ntotal, ngood, untilnbad)
            == mp.ninf)
    assert (negative_hypergeometric.logpmf(sup[-1] + 1, ntotal, ngood, untilnbad)
            == mp.ninf)


@mp.workdps(20)
def test_logpmf_agains_r():
    # See the comments in test_pmf_against_r() for an explanation of
    # the source of the reference values.
    ntotal = 11
    ngood = 6
    untilnbad = 4
    sup = negative_hypergeometric.support(ntotal, ngood, untilnbad)
    logpmf = [negative_hypergeometric.logpmf(k, ntotal, ngood, untilnbad) for k in sup]
    ref = [-4.189654742026, -2.957511060734, -2.223541885654, -1.753538256408,
           -1.481604540924, -1.417066019787, -1.704748092238]
    for logpmf1, ref1 in zip(logpmf, ref):
        assert mp.almosteq(logpmf1, ref1, rel_eps=1e-8, abs_eps=0)


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


def test_cdf_outside_support():
    ntotal = 25
    ngood = 20
    untilnbad = 5
    sup = negative_hypergeometric.support(ntotal, ngood, untilnbad)
    assert negative_hypergeometric.cdf(sup[0] - 1, ntotal, ngood, untilnbad) == 0
    assert negative_hypergeometric.cdf(sup[-1], ntotal, ngood, untilnbad) == 1


@mp.workdps(40)
def test_sf_against_sum():
    ntotal = 30
    ngood = 18
    untilnbad = 3
    sup, pmf = negative_hypergeometric.support_pmf(ntotal, ngood, untilnbad)
    sf_by_sum = list(accumulate(pmf[::-1], initial=0))[-2::-1]
    for k in sup:
        sf = negative_hypergeometric.sf(k, ntotal, ngood, untilnbad)
        assert mp.almosteq(sf, sf_by_sum[k])


def test_sf_outside_support():
    ntotal = 26
    ngood = 21
    untilnbad = 4
    sup = negative_hypergeometric.support(ntotal, ngood, untilnbad)
    assert negative_hypergeometric.sf(sup[0] - 1, ntotal, ngood, untilnbad) == 1
    assert negative_hypergeometric.sf(sup[-1], ntotal, ngood, untilnbad) == 0


@pytest.mark.parametrize('ntotal, ngood, untilnbad',
                         [(32, 24, 4), (200, 125, 17)])
@mp.workdps(50)
def test_mean(ntotal, ngood, untilnbad):
    m = negative_hypergeometric.mean(ntotal, ngood, untilnbad)
    # Compute a reference value using the definition of the mean.
    sup, pmf = negative_hypergeometric.support_pmf(ntotal, ngood, untilnbad)
    ref = mean(sup, weights=pmf)
    assert mp.almosteq(m, ref)


@pytest.mark.parametrize('ntotal, ngood, untilnbad',
                         [(33, 24, 4), (199, 125, 15)])
@mp.workdps(50)
def test_var(ntotal, ngood, untilnbad):
    v = negative_hypergeometric.var(ntotal, ngood, untilnbad)
    # Compute a reference value using the definition of the mean.
    mu = negative_hypergeometric.mean(ntotal, ngood, untilnbad)
    sup, pmf = negative_hypergeometric.support_pmf(ntotal, ngood, untilnbad)
    squared_diffs = [(k - mu)**2 for k in sup]
    ref = mean(squared_diffs, weights=pmf)
    assert mp.almosteq(v, ref)
