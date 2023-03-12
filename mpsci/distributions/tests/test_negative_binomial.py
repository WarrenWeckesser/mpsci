import pytest
from mpmath import mp
from mpsci.distributions import negative_binomial


# In the following, the Wolfram Alpha distribution NegativeBinomialDistribution
# is used as an independent reference.  Wolfram uses a different convention for
# the parameters of the distribution, which amounts to flipping the definition
# of the probability p to 1 - p.  That is, for example,
#
#    PMF[NegativeBinomialDistribution[r, p], k] ==
#        mpsci.distributions.negative_binomial.pmf(k, r, 1 - p)

@pytest.mark.parametrize(
    'k, r, p, ref',
    [(3, 5, 0.25, '8505/65536'),
     (10, 6, 0.125, '353_299_947/281_474_976_710_656'),
     (10, 2.5, 0.125, '1.8921030096242071800154196646369236744610e-8')]
)
@mp.workdps(40)
def test_pmf(k, r, p, ref):
    pmf = negative_binomial.pmf(k, r, p)
    assert mp.almosteq(pmf, mp.mpf(ref))


@pytest.mark.parametrize(
    'k, r, p, ref',
    [(3, 8, 0.625, '20_726_199/1_073_741_824'),
     (10, 2.5, 0.125, '0.999999996873254423162636297285363712912154')]
)
@mp.workdps(40)
def test_cdf(k, r, p, ref):
    cdf = negative_binomial.cdf(k, r, p)
    assert mp.almosteq(cdf, mp.mpf(ref))


@pytest.mark.parametrize(
    'k, r, p, ref',
    [(3, 8, 0.625, '1_053_015_625/1_073_741_824'),
     (10, 2.5, 0.125, '3.12674557683736370271463628708784568429057e-9')]
)
@mp.workdps(40)
def test_sf(k, r, p, ref):
    sf = negative_binomial.sf(k, r, p)
    assert mp.almosteq(sf, mp.mpf(ref))


@mp.workdps(40)
def test_mean():
    r = 12.5
    p = 0.125
    mean = negative_binomial.mean(r, p)
    assert mp.almosteq(mean, mp.mpf('25/14'))


@mp.workdps(40)
def test_var():
    r = 12.5
    p = 0.125
    var = negative_binomial.var(r, p)
    assert mp.almosteq(var, mp.mpf('100/49'))
