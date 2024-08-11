from mpmath import mp
from mpsci.distributions import t


@mp.workdps(50)
def test_pdf():
    df = 10
    x = 5
    pdf = t.pdf(x, df)
    # Expected value confirmed with Wolfram Alpha:
    #     PDF[StudentTDistribution[10], 5]
    expected = 9*mp.sqrt(mp.mpf(5)/7)/19208
    assert mp.almosteq(pdf, expected)


@mp.workdps(50)
def test_cdf():
    df = 10
    x = 5
    cdf = t.cdf(x, df)
    # Expected value confirmed with Wolfram Alpha:
    #     CDF[StudentTDistribution[10], 5]
    expected = mp.one/2 + 3245*mp.sqrt(mp.mpf(5)/7)/5488
    assert mp.almosteq(cdf, expected)


@mp.workdps(50)
def test_cdf_limits():
    df = 10
    cdf = t.cdf(mp.ninf, df)
    assert cdf == 0
    cdf = t.cdf(mp.inf, df)
    assert cdf == 1


@mp.workdps(50)
def test_sf():
    df = 10
    x = 5
    sf = t.sf(x, df)
    # Expected value confirmed with Wolfram Alpha:
    #     1 - CDF[StudentTDistribution[10], 5]
    expected = mp.one/2 - 3245*mp.sqrt(mp.mpf(5)/7)/5488
    assert mp.almosteq(sf, expected)


@mp.workdps(50)
def test_sf_limits():
    df = 10
    sf = t.sf(mp.ninf, df)
    assert sf == 1
    sf = t.sf(mp.inf, df)
    assert sf == 0


@mp.workdps(50)
def test_invcdf_cdf_roundtrip():
    df = 13
    p0 = mp.one/8
    x = t.invcdf(p0, df)
    p1 = t.cdf(x, df)
    assert mp.almosteq(p1, p0)


@mp.workdps(50)
def test_invsf_sf_roundtrip():
    df = 13
    p0 = mp.one/8
    x = t.invsf(p0, df)
    p1 = t.sf(x, df)
    assert mp.almosteq(p1, p0)
