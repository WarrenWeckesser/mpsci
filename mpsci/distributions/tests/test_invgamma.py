from mpmath import mp
from mpsci.distributions import invgamma


@mp.workdps(50)
def test_pdf():
    x = 2
    a = 5
    scale = 3
    p = invgamma.pdf(x, a, scale=scale)
    # Reference value is the formula reported by Wolfram Alpha with the
    # input
    #    PDF[InverseGammaDistribution[5, 3], 2]
    ref = mp.mpf(81)/512*mp.exp(-1.5)
    assert mp.almosteq(p, ref)


@mp.workdps(50)
def test_cdf():
    x = 2
    a = 5
    scale = 3
    cdf = invgamma.cdf(x, a, scale=scale)
    # Reference value is the formula reported by Wolfram Alpha with the
    # input
    #    CDF[InverseGammaDistribution[5, 3], 2]
    ref = mp.mpf(563)/128*mp.exp(-1.5)
    assert mp.almosteq(cdf, ref)


@mp.workdps(50)
def test_sf():
    x = 2
    a = 5
    scale = 3
    sf = invgamma.sf(x, a, scale=scale)
    # Reference value is the formula reported by Wolfram Alpha with the
    # input
    #    1 - CDF[InverseGammaDistribution[5, 3], 2]
    ref = mp.one - mp.mpf(563)/128*mp.exp(-1.5)
    assert mp.almosteq(sf, ref)


@mp.workdps(50)
def test_mean():
    a = 5
    scale = 3
    m = invgamma.mean(a, scale=scale)
    assert m == mp.mpf(3)/4


@mp.workdps(50)
def test_var():
    a = 5
    scale = 3
    v = invgamma.var(a, scale=scale)
    assert v == mp.mpf(3)/16


@mp.workdps(50)
def test_skewness():
    a = 5
    scale = 3
    sk = invgamma.skewness(a, scale=scale)
    # Wolfram Alpha: Skewness[InverseGammaDistribution[5, 3]]
    ref = 2*mp.sqrt(3)
    assert mp.almosteq(sk, ref)


@mp.workdps(50)
def test_kurtosis():
    a = 5
    scale = 3
    sk = invgamma.kurtosis(a, scale=scale)
    # Wolfram Alpha: Kurtosis[InverseGammaDistribution[5, 3]] - 3
    ref = 42
    assert mp.almosteq(sk, ref)
