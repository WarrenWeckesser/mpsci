from mpmath import mp
from mpsci.distributions import betabinomial


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
