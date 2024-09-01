from itertools import product
import pytest
from mpmath import mp
from mpsci.distributions import betabinomial, Initial
from mpsci.stats import unique_counts


@mp.workdps(50)
def test_support():
    n = 5
    a = 2.5
    b = 3.0
    sup = betabinomial.support(n, a, b)
    pmfsum = [betabinomial.pmf(k, n, a, b) for k in sup]
    assert mp.almosteq(sum(pmfsum), 1.0)


@mp.workdps(50)
def test_pmf():
    n = 10
    a = 2
    b = 3
    k = 4
    pmf = betabinomial.pmf(k, n, a, b)
    # Expected value is from Wolfram Alpha:
    #   PDF[BetaBinomialDistribution[2, 3, 10], 4]
    assert mp.almosteq(pmf, mp.mpf(20)/143)


@mp.workdps(50)
def test_cdf():
    n = 10
    a = 2
    b = 3
    k = 4
    prob = betabinomial.cdf(k, n, a, b)
    # Expected value is from Wolfram Alpha:
    #   CDF[BetaBinomialDistribution[2, 3, 10], 4]
    assert mp.almosteq(prob, mp.mpf(85)/143)


@mp.workdps(50)
def test_sf():
    n = 10
    a = 2
    b = 3
    k = 4
    prob = betabinomial.sf(k, n, a, b)
    # Expected value is from Wolfram Alpha:
    #   1 - CDF[BetaBinomialDistribution[2, 3, 10], 4]
    assert mp.almosteq(prob, mp.mpf(58)/143)


@mp.workdps(50)
def test_mean():
    n = 10
    a = 2
    b = 3
    m = betabinomial.mean(n, a, b)
    # Optimistically expect mpmath to get the exact result.
    assert m == 4.0


@mp.workdps(50)
def test_var():
    n = 10
    a = 2
    b = 3
    v = betabinomial.var(n, a, b)
    # Optimistically expect mpmath to get the exact result.
    assert v == 6.0


@mp.workdps(50)
def test_skewness():
    n = 10
    a = 2
    b = 3
    sk = betabinomial.skewness(n, a, b)
    assert mp.almosteq(sk, mp.mpf(5)/(7*mp.sqrt(6)))


@mp.workdps(50)
def test_kurtosis():
    n = 10
    a = 2
    b = 3
    exckurt = betabinomial.kurtosis(n, a, b)
    # Expected value from Wolfram Alpha:
    #   ExcessKurtosis[BetaBinomialDistribution[2, 3, 10]]
    assert mp.almosteq(exckurt, -mp.mpf(29)/42)


@pytest.mark.parametrize(
    'x, n',
    [([1, 1, 2, 4, 8, 15, 16, 16], 20),
     ([1, 1, 5, 5, 5, 2, 3, 3, 5, 8, 13, 21], 21)],
)
@mp.workdps(50)
def test_nll_counts(x, n):
    nll1 = betabinomial.nll(x, n=n, a=1.5, b=3.0)
    xvalues, xcounts = unique_counts(x)
    nll2 = betabinomial.nll(xvalues, counts=xcounts, n=n, a=1.5, b=3.0)
    assert mp.almosteq(nll1, nll2)


@pytest.mark.parametrize('use_counts', [False, True])
@mp.workdps(80)
def test_mle_special_case(use_counts):
    # Experiments show that for a dataset of the form
    #   [0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, ...]
    # (i.e. the kth integer occurs k+1 times), and with n equal to the last
    # integer included in the dataset, the MLE is a=2, b=1. (TODO: Prove this!)
    x = [0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4]
    counts = None
    if use_counts:
        x, counts = unique_counts(x)

    n = 4
    nhat, ahat, bhat = betabinomial.mle(x, n=n, a=Initial(2), counts=counts)
    assert nhat == n
    assert mp.almosteq(ahat, 2)
    assert mp.almosteq(bhat, 1)


@pytest.mark.parametrize(
    'x, n',
    [([2, 4, 8, 16], 20),
     ([1, 1, 2, 3, 5, 8, 13, 21], 21)],
)
@pytest.mark.parametrize('use_counts', [False, True])
@mp.workdps(50)
def test_mle_a_and_b(x, n, use_counts):
    # This is a crude test of the mle() function.
    counts = None
    if use_counts:
        x, counts = unique_counts(x)
    nhat, ahat, bhat = betabinomial.mle(x, n=n, counts=counts)
    assert nhat == n
    nll = betabinomial.nll(x, n=n, a=ahat, b=bhat, counts=counts)
    delta = 1e-9
    nd = 2
    dirs = set(product(*([[-1, 0, 1]]*nd))) - set([(0,)*nd])
    for d in dirs:
        a = ahat + d[0]*delta
        b = bhat + d[1]*delta
        assert nll < betabinomial.nll(x, n=n, a=a, b=b, counts=counts)


@pytest.mark.parametrize(
    'x, n, b',
    [([2, 4, 8, 16], 20, 2.5),
     ([1, 1, 2, 3, 5, 8, 13, 21], 21, 1.25),],
)
@pytest.mark.parametrize('use_counts', [False, True])
@mp.workdps(50)
def test_mle_b_fixed(x, n, b, use_counts):
    # This is a crude test of the mle() function.
    counts = None
    if use_counts:
        x, counts = unique_counts(x)
    nhat, ahat, bhat = betabinomial.mle(x, n=n, b=b, counts=counts)
    assert nhat == n
    assert bhat == b
    nll = betabinomial.nll(x, n=n, a=ahat, b=b, counts=counts)
    delta = 1e-9
    for delta in [-1e-9, 1e-9]:
        a = ahat + delta
        assert nll < betabinomial.nll(x, n=n, a=a, b=b, counts=counts)


@pytest.mark.parametrize(
    'x, n, a',
    [([2, 4, 8, 16], 20, 1.25),
     ([1, 1, 2, 3, 5, 8, 13, 21], 21, 0.6),],
)
@pytest.mark.parametrize('use_counts', [False, True])
@mp.workdps(50)
def test_mle_a_fixed(x, n, a, use_counts):
    # This is a crude test of the mle() function.
    counts = None
    if use_counts:
        x, counts = unique_counts(x)
    nhat, ahat, bhat = betabinomial.mle(x, counts=counts, n=n, a=a)
    assert nhat == n
    assert ahat == a
    nll = betabinomial.nll(x, n=n, a=a, b=bhat, counts=counts)
    delta = 1e-9
    for delta in [-1e-9, 1e-9]:
        b = bhat + delta
        assert nll < betabinomial.nll(x, n=n, a=a, b=b, counts=counts)


@mp.workdps(50)
def test_mle_a_and_b_fixed():
    x = [1, 1, 2, 3, 5, 8, 13]
    n = 15
    a = 1.5
    b = 5
    nhat, ahat, bhat = betabinomial.mle(x, n=n, a=a, b=b)
    assert nhat == n
    assert ahat == a
    assert bhat == b

    xvalues, xcounts = unique_counts(x)
    nhat2, ahat2, bhat2 = betabinomial.mle(xvalues, counts=xcounts,
                                           n=n, a=a, b=b)
    assert nhat2 == n
    assert ahat2 == a
    assert bhat2 == b
