from mpmath import mp
from mpsci.distributions import studentt


@mp.workdps(50)
def test_pdf():
    df = 10
    x = 5
    pdf = studentt.pdf(x, df)
    # Expected value confirmed with Wolfram Alpha:
    #     PDF[StudentTDistribution[10], 5]
    expected = 9*mp.sqrt(mp.mpf(5)/7)/19208
    assert mp.almosteq(pdf, expected)


@mp.workdps(50)
def test_cdf():
    df = 10
    x = 5
    cdf = studentt.cdf(x, df)
    # Expected value confirmed with Wolfram Alpha:
    #     CDF[StudentTDistribution[10], 5]
    expected = mp.one/2 + 3245*mp.sqrt(mp.mpf(5)/7)/5488
    assert mp.almosteq(cdf, expected)


@mp.workdps(50)
def test_cdf_limits():
    df = 10
    cdf = studentt.cdf(mp.ninf, df)
    assert cdf == 0
    cdf = studentt.cdf(mp.inf, df)
    assert cdf == 1


@mp.workdps(50)
def test_sf():
    df = 10
    x = 5
    sf = studentt.sf(x, df)
    # Expected value confirmed with Wolfram Alpha:
    #     1 - CDF[StudentTDistribution[10], 5]
    expected = mp.one/2 - 3245*mp.sqrt(mp.mpf(5)/7)/5488
    assert mp.almosteq(sf, expected)


@mp.workdps(50)
def test_sf_limits():
    df = 10
    sf = studentt.sf(mp.ninf, df)
    assert sf == 1
    sf = studentt.sf(mp.inf, df)
    assert sf == 0


@mp.workdps(50)
def test_invcdf_cdf_roundtrip():
    df = 13
    p0 = mp.one/8
    x = studentt.invcdf(p0, df)
    p1 = studentt.cdf(x, df)
    assert mp.almosteq(p1, p0)


@mp.workdps(50)
def test_invsf_sf_roundtrip():
    df = 13
    p0 = mp.one/8
    x = studentt.invsf(p0, df)
    p1 = studentt.sf(x, df)
    assert mp.almosteq(p1, p0)
