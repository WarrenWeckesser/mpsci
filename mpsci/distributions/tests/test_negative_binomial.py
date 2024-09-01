import pytest
from mpmath import mp
from mpsci.distributions import negative_binomial
from mpsci.stats import unique_counts
from ._utils import call_and_check_mle


def test_support():
    sup = negative_binomial.support(3, 0.25)
    assert next(sup) == 0
    assert next(sup) == 1
    assert next(sup) == 2
    assert 1000 in sup


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


@pytest.mark.parametrize(
    'x, r, p',
    [([1, 1, 2, 4, 8, 15, 16, 24], 1.5, 0.25),
     ([1, 1, 1, 5, 5, 2, 3, 3, 5, 8, 13, 15], 3.0, 0.125)],
)
@mp.workdps(50)
def test_nll_counts(x, r, p):
    nll1 = negative_binomial.nll(x, r=r, p=p)
    xvalues, xcounts = unique_counts(x)
    nll2 = negative_binomial.nll(xvalues, counts=xcounts, r=r, p=p)
    assert mp.almosteq(nll1, nll2)


@mp.workdps(25)
@pytest.mark.parametrize('x', [[0, 1, 2, 3, 5, 8, 13],
                               [0]*155 + [1]*39 + [2]*6 + [3]*1])
def test_mle_basic(x):
    call_and_check_mle(negative_binomial.mle, negative_binomial.nll, x)


@pytest.mark.parametrize('use_counts', [False, True])
@mp.workdps(25)
def test_mle_fixed_r(use_counts):
    # Note: I haven't tried to verify this, but the numerical result
    # suggests that with r fixed to be 5, the MLE for p is 5/6.
    x = [21, 29, 19, 10, 33, 28, 16, 21, 17, 22, 31, 53]
    counts = None
    if use_counts:
        x, counts = unique_counts(x)
    call_and_check_mle(
        lambda x: negative_binomial.mle(x, r=5, counts=counts)[1:],
        lambda x, p: negative_binomial.nll(x, r=5, p=p, counts=counts),
        x
    )


@pytest.mark.parametrize('use_counts', [False, True])
@mp.workdps(25)
def test_mle_fixed_p(use_counts):
    x = [21, 29, 19, 10, 33, 28, 16, 21, 17, 22, 31, 53]
    counts = None
    if use_counts:
        x, counts = unique_counts(x)
    call_and_check_mle(
        lambda x: negative_binomial.mle(x, p=0.75, counts=counts)[:1],
        lambda x, r: negative_binomial.nll(x, r=r, p=0.75, counts=counts),
        x
    )
